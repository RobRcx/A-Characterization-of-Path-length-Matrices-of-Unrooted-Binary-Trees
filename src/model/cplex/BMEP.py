# from docplex.mp.model import Model
from enum import Enum
import numpy as np
import itertools
from os.path import join
import os
import inspect
from docplex.mp.model import Model
from docplex.util.environment import get_environment
from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.mp.relaxer import Relaxer

from util import functions
from util import constants
from model.cplex.CplexSolver import SolverIO, CplexSolver

class ObjType(Enum):
    classic = "BMEP_classic"
    experimental = "BMEP_experimental"
    zero = "zero"

class BMEPSolverIO(SolverIO):
    def __init__(self, instance, obj_type, config_name):
        super().__init__()

        self.set_input_path(constants.Path.instance_path)

        path = join(constants.Path.new_solutions_output_path, join(obj_type.value, config_name))
        if not os.path.exists(path):
            os.makedirs(path)

        self.output_folder_path = path
        lp_out_path = join(self.output_folder_path, f"{instance}_sol")
        txt_out_path = join(self.output_folder_path, f"{instance}_sol.txt")
        summary_out_path = join(self.output_folder_path, "summary.txt")
        self.set_output_paths(lp_out_path, txt_out_path, summary_out_path)


class BMEPSolver(CplexSolver):
    def __init__(self, model_name, n, m, obj_type=ObjType.classic, solverIO=None,
                 buneman_disjunctive=True,       # Apply Buneman's disjunctive constraints
                 manifold=False,                 # Apply UBT-manifold
                 time_limit=None,                # Maximum time limit for computation
                 mipgap=0.0001,                  # Maximum optimality gap w.r.t. best relaxation
                 buneman_quadruplets=None,       # Custom buneman quadruplets
                 circular_orders=None,           # Custom circular orders
                 buneman_violation=False,
                 entries_equal_to_two=None,
                 order_first_row=False,
                 fast_branching=False):
        super().__init__(time_limit=time_limit, mipgap=mipgap)

        # Initialize model parameters
        self.n = n
        self.d = m
        self.model_name = model_name

        # Add objectives of BMEPSolver
        self.solverObjectiveType.add_objective(ObjType)

        # Recognize objective
        if not self.solverObjectiveType.is_valid_objective(obj_type.value):
            functions.error_exit(f"Unrecognized objective function.", -1,
                                 invoking_script=inspect.getfile(inspect.currentframe()))
        self.obj_type = obj_type
        # Select objective function accordingly
        if self.obj_type.value == ObjType.classic.value:
            self.obj_func = self.bmep_objective
        elif self.obj_type.value == ObjType.experimental.value:
            self.obj_func = self.surrogate_bmep_objective
            self.d2 = functions.compute_PLDM(self.d)
        elif self.obj_type.value == ObjType.zero.value:
            self.obj_func = self.zero_objective

        # Add input/output path
        if solverIO is None:
            self.solverIO = BMEPSolverIO(model_name, obj_type)
        else:
            self.solverIO = solverIO

        # Add runtime parameters
        self.solverOptions.ManifoldConstraint = manifold
        self.solverOptions.BunemanDisjunctiveConstraint = buneman_disjunctive

        self.buneman_quadruplets = buneman_quadruplets
        self.circular_orders = circular_orders
        self.buneman_violation = buneman_violation
        self.entries_equal_to_two = entries_equal_to_two
        self.order_first_row = order_first_row
        self.fast_branching = fast_branching

    def bmep_objective(self, x):
        # \sum_{i, j \in [n]}, i \neq j} d_{i, j} ( \sum_{l \in [2, n - 1]} 2^{-l} x_{i, j}^l )
        n = self.n
        return self.model.sum([self.d[i - 1][j - 1] * x[i, j, l] * (2 ** (-l))
                               for l in range(2, n) for j in range(1, n + 1) for i in range(1, n + 1)])

    def surrogate_bmep_objective(self, x):
        # \sum_{i, j \in [n]}, i \neq j} tau_{i, j} 2^{-d_{i, j}}
        n = self.n
        return self.model.sum([l * x[i, j, l] * self.d2[i - 1][j - 1]
                               for l in range(2, n) for j in range(1, n + 1) for i in range(1, n + 1)])

    def zero_objective(self, x):
        return 0

    def build_model(self):
        print("------------ Building model with Integrality, Symmetry, Zero diagonal, Kraft and Strong Triangle inequalities.")

        self.model = Model(self.model_name)

        # Alias
        n = self.n
        model = self.model

        # Set execution parameters saved in the constructor
        if self.time_limit is not None:
            model.time_limit = self.time_limit
        model.parameters.mip.tolerances.mipgap = self.mipgap

        #####################################################################################
        ############################## DECISION VARIABLES ###################################
        #####################################################################################
        # x_{i, j}^l, \forall i, j \in [n], i \neq j, l \in [2, n - 1]
        # x = model.binary_var_cube(keys1=range(1, n + 1), keys2=range(1, n + 1), keys3=range(1, n + 1), name='x')
        x = model.binary_var_dict(((i, j, k) for i in range(1, n + 1)
                                 for j in range(1, n + 1)
                                 for k in range(2, n)), name="x")

        # y_{p, q}^j, \forall j, p, q \in [n], i \neq j \neq p \neq q
        if self.solverOptions.BunemanDisjunctiveConstraint and not self.buneman_violation:
            y = model.binary_var_cube(keys1=range(1, n + 1), keys2=range(1, n + 1), keys3=range(1, n + 1), name='y')
        # \tau_{i, j} \in [2, n - 1] \forall i, j \in [n], i \neq j
        tau = model.continuous_var_matrix(keys1=range(1, n + 1), keys2=range(1, n + 1), name='tau')


        ##############################################################################
        ############################## OBJECTIVE ###################################
        ##############################################################################
        self.model.minimize(self.obj_func(x))


        ##############################################################################
        ############################## CONSTRAINTS ###################################
        ##############################################################################

        ############################## NULL DIAGONAL ###################################
        # \sum_{l \in [2, n - 1]} x_{i, j}^l = 1  \forall i, j \in [n], i \neq j
        # model.add_constraints(model.sum((x[i, j, l] if i != j else 0) for l in range(2, n)) == 1
        #                                for j in range(1, n + 1) for i in range(1, n + 1))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    model.add_constraint(model.sum(x[i, j, l] for l in range(2, n)) == 1, f"(27b)i={i}j={j}")
                else:
                    for l in range(2, n):
                        model.add_constraint(x[i, j, l] == 0)

        ############################## SYMMETRY ###################################
        # x_{i, j}^l = x_{j, i}^l  \forall i, j \in [n], i \neq j, l \in [2, n - 1]
        # model.add_constraints((x[i, j, l] == x[j, i, l] if i != j else tau[1, 1] == tau[1, 1])
        #                      for l in range(2, n) for j in range(1, n + 1) for i in range(1, n + 1))
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                for l in range(2, n):
                    if i != j:
                        model.add_constraint(x[i, j, l] == x[j, i, l], f"(27c)i={i}j={j}l={l}")

        ############################## TIE tau and x ###################################
        # \tau_{i, j} = \sum_{l \in [2, n - 1]} l x_{i, j}^l  \forall i, j \in [n], i \neq j
        # model.add_constraints(tau[i, j] == model.sum(l * x[i, j, l] for l in range(2, n))
        #                      for j in range(1, n + 1) for i in range(1, n + 1))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                # if i != j:
                model.add_constraint(tau[i, j] == model.sum(l * x[i, j, l] for l in range(2, n)), f"(27c)i={i}j={j}")

        ############################## KRAFT ###################################
        # \sum_{j \in [n]\{i}} \sum_{l \in [2, n - 1]} {2^{-l} x_{i, j}^l} = 1 / 2  \forall i \in [n]
        model.add_constraints(model.sum((2 ** (-l)) * x[i, j, l] if j != i else 0 for l in range(2, n) for j in range(1, n + 1)) == 1 / 2
                                            for i in range(1, n + 1))

        ############################## STRONG TRIANGLE INEQUALITIES ###################################
        # \tau_{i, j} + \tau_{j, k} - \tau_{i, k} >= 2  \forall i, j, k \in [n] : i \neq j \neq k
        model.add_constraints(
            (tau[i, j] + tau[j, k] - tau[i, k] >= 2 if i != j and j != k and i != k else tau[1, 1] == tau[1, 1])
            for k in range(1, n + 1) for j in range(1, n + 1) for i in range(1, n + 1))

        ############################## BUNEMAN DISJUNCTIVE ###################################
        if self.solverOptions.BunemanDisjunctiveConstraint and not self.buneman_violation:
            print("--> Activating Buneman disjunctive constraints.")
            # .............................  \forall i, j, p, q \in [n] : i \neq j \neq p \neq q
            model.add_constraints(
                (y[p, q, j] + y[j, q, p] + y[j, p, q] == 1 if j != p and p != q and j != q else tau[1, 1] == tau[1, 1])
                for q in range(2, n + 1) for p in range(2, n + 1) for j in range(2, n + 1))
            # Strong four-point conditions
            # d_{i, j} + d_{p, q} <= max(d_{i, p} + d{j, q}, d_{j, p} + d_{i, q})
            model.add_constraints(
                tau[1, j] + tau[p, q] >= 2 * (1 - y[j, p, q]) + tau[1, p] + tau[j, q] - (2 * n - 2) * y[p, q, j]
                if j != p and p != q and j != q else tau[1, 1] == tau[1, 1]
                for q in range(2, n + 1) for p in range(2, n + 1) for j in range(2, n + 1))
            model.add_constraints(
                tau[1, p] + tau[j, q] >= 2 * (1 - y[j, p, q]) + tau[1, j] + tau[p, q] - (2 * n - 2) * y[j, q, p]
                if j != p and p != q and j != q else tau[1, 1] == tau[1, 1]
                for q in range(2, n + 1) for p in range(2, n + 1) for j in range(2, n + 1))
            model.add_constraints(
                tau[1, j] + tau[p, q] >= 2 * (1 - y[j, q, p]) + tau[1, q] + tau[j, p] - (2 * n - 2) * y[p, q, j]
                if j != p and p != q and j != q else tau[1, 1] == tau[1, 1]
                for q in range(2, n + 1) for p in range(2, n + 1) for j in range(2, n + 1))
            model.add_constraints(
                tau[1, q] + tau[j, p] >= 2 * (1 - y[j, q, p]) + tau[1, j] + tau[p, q] - (2 * n - 2) * y[j, p, q]
                if j != p and p != q and j != q else tau[1, 1] == tau[1, 1]
                for q in range(2, n + 1) for p in range(2, n + 1) for j in range(2, n + 1))
            model.add_constraints(
                tau[1, p] + tau[j, q] >= 2 * (1 - y[p, q, j]) + tau[1, q] + tau[j, p] - (2 * n - 2) * y[j, q, p]
                if j != p and p != q and j != q else tau[1, 1] == tau[1, 1]
                for q in range(2, n + 1) for p in range(2, n + 1) for j in range(2, n + 1))
            model.add_constraints(
                tau[1, q] + tau[j, p] >= 2 * (1 - y[p, q, j]) + tau[1, p] + tau[j, q] - (2 * n - 2) * y[j, p, q]
                if j != p and p != q and j != q else tau[1, 1] == tau[1, 1]
                for q in range(2, n + 1) for p in range(2, n + 1) for j in range(2, n + 1))

        ############################## UBT-MANIFOLD ###################################
        if self.solverOptions.ManifoldConstraint:
            print("--> Activating Manifold constraint.")
            model.add_constraint(model.sum(l * (2 ** (-l)) * x[i, j, l] if j != i else 0
                                            for l in range(2, n) for j in range(1, n + 1) for i in range(1, n + 1)) == 2 * n - 3)

        ############################## CUSTOM BUNEMAN ###################################
        if self.buneman_quadruplets is not None:
            print("--> Activating custom Buneman constraints.")
            for idx in range(len(self.buneman_quadruplets)):
                v = self.buneman_quadruplets[idx]
                for x in v:
                    i, j, p, q = x
                    if idx == 0:
                        model.add_constraint(tau[i, q] + tau[j, p] == tau[i, p] + tau[j, q])
                    elif idx == 1:
                        model.add_constraint(tau[i, j] + tau[p, q] == tau[i, p] + tau[j, q])
                    elif idx == 2:
                        model.add_constraint(tau[i, j] + tau[p, q] == tau[i, q] + tau[j, p])

        ############################## CUSTOM CIRCULAR ORDERS ###################################
        if self.circular_orders is not None:
            print(f"--> Activating {len(self.circular_orders)} custom circular orders.")
            for seq in self.circular_orders:
                # print(seq)
                model.add_constraint(model.sum(tau[seq[j - 1], seq[j]] for j in range(1, n)) >= 4 * n - 8)

        ############################## BUNEMAN VIOLATION ###################################
        if self.buneman_violation:
            print("--> Activating Buneman violation constraints.")
            model.add_constraint(tau[1, 2] + tau[3, 4] + 2 >= tau[1, 3] + tau[2, 4] + 1, "bv1")  # model.add_constraint((tau[1, 3] + tau[2, 4]) != (tau[1, 2] + tau[3, 4]))
            model.add_constraint(tau[1, 3] + tau[2, 4] + 2 >= tau[1, 2] + tau[3, 4] + 1, "bv2")
            model.add_constraint(tau[1, 4] + tau[2, 3] + 2 >= tau[1, 2] + tau[3, 4] + 1, "bv3")

        ############################## SYMMETRY REDUCTION WHEN GIVEN NUMBER OF ENTRIES = 2 ###################################
        if self.entries_equal_to_two is not None:
            print(f"--> Imposing number of entries equal to 2 in upper triangular matrix equal to {self.entries_equal_to_two}.")
            model.add_constraint(model.sum(x[i, j, 2] for i in range(1, n + 1) for j in range(i + 1, n + 1)) == self.entries_equal_to_two,
                                 f"(cherry)i={i}j={j}")

            '''print(f"--> Imposing number of entries equal to n - 1 in upper triangular matrix less than or equal to 2.")
            model.add_constraint(model.sum(x[i, j, n - 1] for i in range(1, n + 1) for j in range(i + 1, n + 1)) <= 2,
                                 f"(caterpillar)i={i}j={j}")'''



            if self.entries_equal_to_two == 2:
                print(f"--> Imposing that, since (number_of_entries_equal_to_2) = 2, then: \n"
                      f"\t\tthe number of entries equal to n - 1 is 4, "
                      f"\t\tthe number of entries equal to n - 2 is 4, \n"
                      f"\t\tthe number of entries equal to n - 3 is 5, "
                      f"\t\t..., \n"
                      f"\t\tthe number of entries equal to 3 is n - 1.")
                model.add_constraint(model.sum(x[i, j, n - 1]for i in range(1, n + 1) for j in range(i + 1, n + 1)) == 4)
                for d in range(4, n):
                    model.add_constraint(model.sum(x[i, j, n - d + 2] for i in range(1, n + 1) for j in range(i + 1, n + 1)) == d)
            '''print(f"--> Imposing that, if (number_of_entries_equal_to_2) = 2, then: \n"
                  f"\t\tthe number of entries equal to n - 1 is = 4, "
                  f"\t\tthe number of entries equal to n - 2 is >= 2, "
                  f"\t\t..., "
                  f"\t\tthe number of entries equal to 3 is >= 2")
            if self.entries_equal_to_two == 2:
                model.add_constraint(model.sum(x[i, j, n - 1] for i in range(1, n + 1) for j in range(i + 1, n + 1)) == 4,
                                     f"(caterpillar)i={i}j={j}l={n - 1}")
                for l in range(3, n - 1):
                    model.add_constraint(
                        model.sum(x[i, j, l] for i in range(1, n + 1) for j in range(i + 1, n + 1)) >= 2,
                        f"(caterpillar)i={i}j={j}l={l}")'''


            if self.entries_equal_to_two == 3:
                print(
                    f"--> Imposing that, since (number_of_entries_equal_to_2) = 3, then:\n"
                    f"\t\tthe number of entries equal to n - 2 is <= 4\n")
                model.add_constraint(model.sum(x[i, j, n - 2] for i in range(1, n + 1) for j in range(i + 1, n + 1)) <= 4)


            #print(f"--> Imposing that the number of entries equal to n - d is equal to 0 "
            #      f"for each d = 1, 2, ..., (number_of_entries_equal_to_2) - 2.")
            print("--> Imposing that:")
            for d in range(1, self.entries_equal_to_two - 2 + 1):
                print(f"\t\tThe number of entries equal to {n - d} is zero.")
                model.add_constraints((x[i, j, n - d] == 0) for i in range(1, n + 1) for j in range(i + 1, n + 1))

        ############################## FIRST ROW NON-DECREASING ORDER ###################################
        if self.order_first_row:
            print(f"--> Activating non-decreasing order on first row.")
            model.add_constraints(tau[1, i] <= tau[1, i + 1] for i in range(1, n))

        '''
        if self.reduce_symmetry:
            # FIRST ALTERNATIVE: set u and z and then minimize sum of u's
            z = model.binary_var_dict(((i, j) for i in range(1, n + 1) for j in range(1, n + 1)), name="z")
            u = model.continuous_var_dict(((i, j) for i in range(1, n + 1) for j in range(1, n + 1)), name="u")
            for j in range(1, n):
                for i in range(1, j):
                    model.add_constraint(u[i, j + 1] >= tau[i, j + 1] - tau[i, j])
                    model.add_constraint(u[i, j + 1] >= tau[i, j] - tau[i, j + 1])
                    model.add_constraint(u[i, j + 1] >= 0)
                    model.add_constraint(z[i, j + 1] <= u[i, j + 1])
            model.minimize()'''
        '''
            # SECOND ALTERNATIVE: disjunctive constraints
            z = model.binary_var_dict(((i, j) for i in range(1, n + 1) for j in range(1, n + 1)), name="z")
            tmp = model.continuous_var_dict(((i, j) for i in range(1, n + 1) for j in range(1, n + 1)), name="u")
            for j in range(1, n):
                for i in range(1, j):
                    model.add_constraint(z[i, j + 1] - n * (1 - tmp[i, j + 1]) <= tau[i, j + 1] - tau[i, j])
                    model.add_constraint(z[i, j + 1] - n * tmp[i, j + 1] <= tau[i, j] - tau[i, j + 1])
            for j in range(1, n):
                for i in range(1, j):
                    model.add_constraint(-n * (model.sum(z[k, j + 1] for k in range(1, i + 1)))
                                              <= tau[i + 1, j + 1] - tau[i + 1, j])'''

        ############################## FAST BRANCHING ###################################
        if self.fast_branching:
            print("--> Activating fast branching.")
            self.set_fast_branching(n, x)

        return model

    def retrieve_solution(self):
        tau = np.zeros((self.n, self.n))
        x = np.zeros((self.n, self.n, self.n))
        y = np.zeros((self.n, self.n, self.n))
        n2v = {"x": x, "y": y, "tau": tau}
        for v in itertools.chain(self.model.iter_binary_vars(),
                                self.model.iter_integer_vars(),
                                self.model.iter_continuous_vars()):
            spl = str(v).split("_")
            name = spl[0]
            idx = np.asarray([int(spl[i]) - 1 for i in range(1, len(spl))])
            if name in n2v:
                n2v[name][tuple(idx)] = v.solution_value
            else:
                continue
            # print(f"{v} = {v.solution_value} = {n2v[name][tuple(idx)]}")
        return tau, x, y

    def export_solution(self, filepath=None):
        tau, _, _ = self.retrieve_solution()
        with open(filepath, "w+") as f:
            f.write(f"{self.n}\n")
            for i in range(self.n):
                for j in range(self.n):
                    f.write(f"{round(tau[i][j])} ")
                f.write("\n")


