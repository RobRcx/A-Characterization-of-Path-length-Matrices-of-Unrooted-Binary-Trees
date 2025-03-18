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
    classic = "BMEP_classic_contractions"
    experimental = "BMEP_experimental_contractions"

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

        # Add input/output path
        if solverIO is None:
            self.solverIO = BMEPSolverIO(model_name, obj_type)

        self.solverOptions.ManifoldConstraint = manifold

    def compute_d2(self, d):
        n = d.shape[0]
        assert n == d.shape[1]
        d2 = np.zeros((n, n))
        for i in range(0, n):
            for j in range(0, n):
                d2[i][j] = 1 / (2 ** d[i][j])
        return d2

    def bmep_objective(self, x):
        return self.model.sum([2 * self.d[i - 1][j - 1] *
                               x[i, j, l] * (2 ** (-l))
                               for l in range(2, self.n) for i in range(1, self.n + 1) for j in range(i + 1, self.n + 1)])

    def surrogate_bmep_objective(self, tau):
        return self.model.sum(2 * tau[i, j, 0] * self.d2[i - 1][j - 1]
                              for i in range(1, self.n + 1) for j in range(i + 1, self.n + 1))

    def build_model(self):
        print("------------ Building model based on contractions.")

        self.model = Model(self.model_name)

        # Alias
        n = self.n
        model = self.model

        # Set execution parameters saved in the constructor
        model.time_limit = self.time_limit
        model.parameters.mip.tolerances.mipgap = self.mipgap

        # Decision variables

        contractions = n - 3
        # tau = model.integer_var_matrix(keys1=range(1, n + 1), keys2=range(1, n + 1), name='tau')
        tau_c = model.continuous_var_cube(keys1=range(1, n + 1), keys2=range(1, n + 1), keys3=range(0, contractions + 1),
                                       name='tau')
        z = model.binary_var_cube(keys1=range(1, n + 1), keys2=range(1, n + 1), keys3=range(1, contractions + 1),
                                  name='z')
        y = model.continuous_var_matrix(keys1=range(1, n + 1), keys2=range(1, contractions + 1), name='y')
        u = model.continuous_var_matrix(keys1=range(1, n + 1), keys2=range(1, contractions + 1), name='u')
        if self.obj_type is ObjType.classic:
            x = model.continuous_var_cube(keys1=range(1, n + 1), keys2=range(1, n + 1), keys3=range(2, n),
                                          name='x')

        # Objective
        if self.obj_type is ObjType.classic:
            model.minimize(self.bmep_objective(x))
        elif self.obj_type is ObjType.experimental:
            model.minimize(self.surrogate_bmep_objective(tau_c))
        else:
            stack = inspect.stack()
            functions.error_exit(message=f"Tried to set unrecognized objective {self.obj_type}",
                                 code=-1,
                                 invoking_script=inspect.getfile(inspect.currentframe()),
                                 line=stack[-1].lineno)

        # Definition of v
        if self.obj_type is ObjType.classic:
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i != j:
                        model.add_constraint(model.sum(x[i, j, l] for l in range(2, n)) == 1)
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    for l in range(2, n):
                        if i != j:
                            model.add_constraint(x[i, j, l] == x[j, i, l])
                        else:
                            model.add_constraint(x[i, j, l] == 0)
            # Linking tau_c and v
            if self.obj_type is ObjType.classic:
                for i in range(1, n + 1):
                    for j in range(i + 1, n + 1):
                        model.add_constraint(tau_c[i, j, 0] == model.sum(l * x[i, j, l] for l in range(2, n)))
        ''' 
        # Kraft
        # if self.obj_type is BMEPObjectiveType.classic_contraction:
        #     model.add_constraints(model.sum((2 ** (-l)) * x[i, j, l] if j != i else 0
        #                                   for l in range(2, n) for j in range(1, n + 1)) == 1 / 2 for i in range(1, n + 1))
        # Triangular
        # model.add_constraints((tau_c[i, j, s] + tau_c[j, k, s] - tau_c[i, k, s] >= 2 if i != j and j != k and i != k else tau_c[1, 1, s] == tau_c[1, 1, s])
        #                                  for k in range(1, n + 1) for j in range(1, n + 1) for i in range(1, n + 1) for s in range(0, contractions + 1))
        '''

        for s in range(1, contractions + 1):
            model.add_constraint(z[n - 2, n - 1, s] == 0)
            model.add_constraint(z[n - 2, n, s] == 0)
            model.add_constraint(z[n - 1, n, s] == 0)

        # The matrices T^s are symmetric
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                for s in range(0, contractions + 1):
                    model.add_constraint(tau_c[i, j, s] == tau_c[j, i, s])

        # The matrices T^s have zero diagonal
        for i in range(1, n + 1):
            for s in range(0, contractions + 1):
                model.add_constraint(tau_c[i, i, s] == 0)

        # Each row/column i \in [1, n - 3] must be involved in a contraction that sets its entries to zero exactly once
        for i in range(1, n - 2):
            model.add_constraint(model.sum(z[i, j, s] for s in range(1, n - 2) for j in range(i + 1, n + 1)) == 1)

        # We perform one contraction at a time
        for s in range(1, contractions + 1):
            model.add_constraint(model.sum(z[i, j, s] for i in range(1, n + 1) for j in range(i + 1, n + 1)) == 1)

        # We can perform the s-th contraction on rows i and j only if tau_ij^{s-1}=2
        for s in range(1, contractions + 1):
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    model.add_constraint(2 * z[i, j, s] <= tau_c[i, j, s - 1])
                    model.add_constraint(tau_c[i, j, s - 1] <= (n - s) + (2 + s - n) * z[i, j, s])

        # The entries of the matrix T^{n - 3} must describe a three-taxon UBT
        model.add_constraint(tau_c[n - 2, n - 1, contractions] == tau_c[n - 2, n, contractions])
        model.add_constraint(tau_c[n - 2, n - 1, contractions] == tau_c[n - 1, n, contractions])
        model.add_constraint(tau_c[n - 2, n - 1, contractions] == 2)

        # y_i^s = 1 if i collapsed at s (=> Taxon i must collapse at most once ?)
        for s in range(1, contractions + 1):
            for i in range(1, n + 1):
                model.add_constraint(y[i, s] == model.sum(z[i, j, s] for j in range(i + 1, n + 1)))
        # u_j^s = 1 if j contracted at s
        for s in range(1, contractions + 1):
            for j in range(1, n + 1):
                model.add_constraint(u[j, s] == model.sum(z[i, j, s] for i in range(1, j)))
        # Once collapsed at s, taxon i cannot be involved in a contraction at any r > s
        for s in range(1, contractions + 1):
            for i in range(1, n + 1):
                model.add_constraint(model.sum(u[i, r] for r in range(s, contractions + 1)) <= (n - s) * (1 - y[i, s]))

        for s in range(1, contractions + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    # model.add_constraint(tau_c[i, j, s] != 1)
                    if i != j:
                        # model.add_constraint(x[i, j, s] >= y[i, s])
                        # model.add_constraint(x[i, j, s] >= y[j, s])
                        # model.add_constraint(y[i, s] + y[j, s] >= x[i, j, s])

                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # model.add_constraint(tau_c[i, j, s] <= (n - 1) * (1 - x[i, j, s]))
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                        model.add_constraint(tau_c[i, j, s] <= tau_c[i, j, s - 1])

                        model.add_constraint(tau_c[i, j, s] <= tau_c[i, j, s - 1] - u[i, s] - u[j, s]
                                             + model.sum(y[i, r] + y[j, r] for r in range(1, s + 1)))
                        # model.add_constraint(tau_c[i, j, s - 1] - u[i, s] - u[j, s]
                        #                      - (n - 1) * (model.sum(x[i, j, r] for r in range(1, s + 1)))
                        #                      <= tau_c[i, j, s])
                        model.add_constraint(tau_c[i, j, s - 1] - u[i, s] - u[j, s]
                                             - (n - 1) * (model.sum(y[i, r] + y[j, r] for r in range(1, s + 1)))
                                             <= tau_c[i, j, s])

        # Symmetry reduction constraints. Each matrix T^s must have at least two entries equal to 2.
        # We perform the contraction on the entry belonging to the row with the lower index:
        '''for s in range(1, n - 2):
            for i in range(1, n - 2):
                for j in range(i + 1, n + 1):
                    for p in range(1, i):
                        for q in range(p + 1, n + 1):
                            model.add_constraint(z[i, j, s] <= tau_c[p, q, s - 1] - 2
                                                 + 3 * model.sum(model.sum(z[u, p, r] for u in range(p + 1, n + 1))
                                                                 + model.sum(z[q, u, r] for u in range(q + 1, n + 1))
                                                                 for r in range(1, s)))'''
        for i in range(1, n + 1):
            for j in range(1, i):
                for s in range(1, contractions + 1):
                    model.add_constraint(z[i, j, s] == 0)

        for s in range(contractions + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    for k in range(1, n + 1):
                        if i != j and i != k and j != k:
                            model.add_constraint(tau_c[i, k, s] - tau_c[j, k, s] >=
                                                 - (n - 3) * (tau_c[i, j, s] - 2)
                                                 - (n - 3) * (model.sum(y[i, r] + y[j, r] + y[k, r] for r in range(1, s + 1))),
                                                 f"nc{s}_{i}_{j}_{k}_ge")
                            model.add_constraint(tau_c[i, k, s] - tau_c[j, k, s] <=
                                                 + (n - 3) * (tau_c[i, j, s] - 2)
                                                 + (n - 3) * (model.sum(y[i, r] + y[j, r] + y[k, r] for r in range(1, s + 1))),
                                                 f"nc{s}_{i}_{j}_{k}_le")

        if self.solverOptions.ManifoldConstraint:
            print("--> Activating Manifold constraint.")
            model.add_constraint(model.sum(l * (2 ** (-l)) * x[i, j, l] if j != i else 0
                                           for l in range(2, n) for j in range(1, n + 1) for i in
                                           range(1, n + 1)) == 2 * n - 3)

        return model

    def retrieve_solution(self):
        v = np.zeros((self.n + 1, self.n + 1, self.n + 1))
        tau = np.zeros((self.n, self.n, self.n - 2))
        z = np.zeros((self.n + 1, self.n + 1, self.n + 1))
        y = np.zeros((self.n + 1, self.n - 2))
        u = np.zeros((self.n + 1, self.n - 2))

        n2v = {"tau": tau, "z": z, "y": y, "u": u, "v": v}
        for v in itertools.chain(self.model.iter_binary_vars(),
                                self.model.iter_integer_vars(),
                                 self.model.iter_continuous_vars()):
            spl = str(v).split("_")
            name = spl[0]
            if name == '':
                continue
            idx = np.asarray([int(spl[i]) for i in range(1, len(spl))])
            # print(f"{name} {idx}  {v.solution_value}")
            if name == "tau":
                tau[idx[0] - 1][idx[1] - 1][idx[2]] = v.solution_value
            elif name in n2v:
                n2v[name][tuple(idx)] = v.solution_value
        return tau, z, y, u, v

    def export_solution(self, filepath=None):
        tau, _, _, _, _ = self.retrieve_solution()
        with open(filepath, "w+") as f:
            f.write(f"{self.n}\n")
            for i in range(self.n):
                for j in range(self.n):
                    f.write(f"{round(tau[i][j][0])} ")
                f.write("\n")