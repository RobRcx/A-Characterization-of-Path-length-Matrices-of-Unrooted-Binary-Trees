from enum import Enum
from os.path import join
import inspect

from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.mp.relaxer import Relaxer
from docplex.mp.relax_linear import LinearRelaxer

from util.constants import DefaultValues, RunOptions
from instance import instance_reader
from util import functions
from ..Solver import Solver, SolverIO, SolverOptions, SolverObjectiveType
from util.constants import SolverMode


class CplexSolver(Solver):
    def __init__(self, time_limit=None, mipgap=0.0001):
        super().__init__(time_limit, mipgap)

    def build_model(self):
        return None

    @staticmethod
    def read_instance(instance_path, filename):
        return instance_reader.read_instance(join(instance_path, filename))

    def set_objective(self, model, rule, sense=1): # sense=1 for minimization
        pass

    def solve(self, model=None, backend="cplex", log_output=False, solver_options=None, integer=True):
        if solver_options is None:
            solver_options = {}
        if model is None:
            stack = inspect.stack()
            functions.error_exit(message="Tried to solve a model that is still to be built.",
                                 code=-1,
                                 invoking_script=inspect.getfile(inspect.currentframe()),
                                 line=stack[-1].lineno)

        # 2. Solve the model
        solution = model.solve(log_output=log_output)  # Solve

        if solution is None:
            return False, None, None

        '''
            Get information on the solution
        '''
        # 3. Get the underlying CPLEX model
        cpx = model.get_engine().get_cplex()

        # 4. Get number of nodes processed
        nodes_processed = cpx.solution.progress.get_num_nodes_processed()

        # 5. Get number of cuts
        #    This returns a tuple of cuts by type; you may sum them if you want total cuts.
        num_cuts = [0]
        if integer:
            for ct in cpx.solution.MIP.cut_type:
                num_cuts.append(cpx.solution.MIP.get_num_cuts(ct))

        res_map = {"obj": solution.get_objective_value(), "nodes": nodes_processed, "cuts": sum(num_cuts)}

        return True, solution, res_map

    def solve_relaxed(self, model, backend="cplex", log_output=False, solver_options=None):
        '''
                TODO: convert
                to
                Pyomo
                from docplex.util.environment import get_environment
                from docplex.mp.conflict_refiner import ConflictRefiner
                from docplex.mp.relaxer import Relaxer

                def refine_conflicts(self):
                    cr = ConflictRefiner()
                    crr = cr.refine_conflict(self.model, display=True)'''
        rx = LinearRelaxer()
        rs = rx.linear_relaxation(model)
        return self.solve(rs, log_output=log_output, integer=False)

    def disable_presolve_cut_heu(self, mdl):
        # -----------------------------------------------------------------
        # 1. Disable Presolve
        # -----------------------------------------------------------------
        # Set the presolve parameter to 0 to turn presolve off
        mdl.parameters.preprocessing.presolve = 0

        # If you also want to disable other preprocessing reductions, you can set:
        mdl.parameters.preprocessing.reduce = 0

        # Disable node presolve
        mdl.parameters.mip.strategy.presolvenode = -1

        # -----------------------------------------------------------------
        # 2. Disable Cuts
        # -----------------------------------------------------------------
        # CPLEX has different cut types (Gomory, MIR, Flow covers, etc.).
        # For each cut type, setting the parameter to 0 disables that cut.
        # The values typically mean:
        #   -1 => automatic (let CPLEX decide)
        #    0 => no generation (disable)
        #    1 => moderate generation
        #    2 => aggressive generation
        # TODO: CORRECT DOCUMENTATION!!!

        mdl.parameters.mip.cuts.covers = -1
        mdl.parameters.mip.cuts.disjunctive = -1
        mdl.parameters.mip.cuts.flowcovers = -1
        mdl.parameters.mip.cuts.gomory = -1
        mdl.parameters.mip.cuts.implied = -1
        mdl.parameters.mip.cuts.liftproj = -1
        mdl.parameters.mip.cuts.mcfcut = -1
        mdl.parameters.mip.cuts.mircut = -1
        mdl.parameters.mip.cuts.pathcut = -1
        mdl.parameters.mip.cuts.zerohalfcut = -1
        mdl.parameters.mip.cuts.cliques = -1
        mdl.parameters.mip.cuts.gubcovers = -1

        # Disable cut passes
        mdl.parameters.mip.limits.cutpasses = -1

        # Disable BQP cuts (if relevant):
        # (Only in newer versions of CPLEX, might not exist in older ones)
        # mdl.parameters.mip.cuts.bqpcuts = 0

        # -----------------------------------------------------------------
        # 3. Disable Heuristics
        # -----------------------------------------------------------------
        # This parameter sets how often (in node count) CPLEX applies its
        # internal MIP heuristics. Setting it to -1 disables them entirely.

        mdl.parameters.mip.strategy.heuristicfreq = -1

        # -----------------------------------------------------------------
        # 4. Disable Aggregator
        # -----------------------------------------------------------------

        mdl.parameters.preprocessing.aggregator = 0

        # -----------------------------------------------------------------
        # 5. Keep Traditional Search (Apply traditional branch and cut strategy; disable dynamic search)
        # -----------------------------------------------------------------

        mdl.parameters.mip.strategy.search = 1

    def refine_conflicts(self, model):
        cr = ConflictRefiner()
        crr = cr.refine_conflict(model, display=True)

    def enable_turbo(self, model):
        model.context.cplex_parameters.preprocessing.presolve = 1
        # Increase number of threads for parallel processing
        model.context.cplex_parameters.threads = 32
        # Set a relative MIP gap tolerance
        model.context.cplex_parameters.mip.tolerances.mipgap = 0.1
        # Adjust heuristic frequency to improve the search for feasible solutions
        model.context.cplex_parameters.mip.strategy.heuristicfreq = 1000
        # You can also adjust emphasis based on your priority
        model.context.cplex_parameters.emphasis.mip = 1

    def select_solver_mode(self, solver, solver_mode):
        if solver_mode.value == SolverMode.Barebone.value:
            solver.disable_presolve_cut_heu(solver.model)
            # print("CPLEX internal preprocessing strategies and cut families disabled.")
        elif solver_mode.value == SolverMode.Turbo.value:
            solver.enable_turbo(solver.model)
            # print("CPLEX high heuristics frequency, high mipgap, high number of threads.")
        elif solver_mode.value == SolverMode.Default.value:
            return
            # print("CPLEX default parameters set.")
        else:
            print("Unrecognized execution mode. Sticking to default parameters.")

    def print_solver_info(self, model):
        print("------------ Solver options\n"
              f"--> Time limit : {model.time_limit}\n"
              f"--> Maximum optimality gap : {model.parameters.mip.tolerances.mipgap}\n"
              f"--> Threads : {model.context.cplex_parameters.threads}\n"
              f"--> Heuristics frequency : {model.context.cplex_parameters.mip.strategy.heuristicfreq} \t(every such number of nodes, CPLEX applies heuristics; -1 heuristics disabled)"
              )
        print("--> Disabled cut families: ", end='')
        # -1 do not generate, 0 default, 1 moderate, 2 aggressive, 3 very aggressive
        # (according to CPLEX ibm.com/docs/en/icos/22.1.0?topic=parameters-mip-covers-switch)
        if model.parameters.mip.cuts.covers.get() == -1: print("covers, ", end='')
        if model.parameters.mip.cuts.disjunctive.get() == -1: print("disjunctive, ", end='')
        if model.parameters.mip.cuts.flowcovers.get() == -1: print("flowcovers, ", end='')
        if model.parameters.mip.cuts.gomory.get() == -1: print("gomory, ", end='')
        if model.parameters.mip.cuts.implied.get() == -1: print("implied, ", end='')
        if model.parameters.mip.cuts.liftproj.get() == -1: print("liftproj, ", end='')
        if model.parameters.mip.cuts.mcfcut.get() == -1: print("mcfcut, ", end='')
        if model.parameters.mip.cuts.mircut.get() == -1: print("mircut, ", end='')
        if model.parameters.mip.cuts.pathcut.get() == -1: print("pathcut, ", end='')
        if model.parameters.mip.cuts.zerohalfcut.get() == -1: print("zerohalfcut, ", end='')
        if model.parameters.mip.cuts.cliques.get() == -1: print("cliques, ", end='')
        if model.parameters.mip.cuts.gubcovers.get() == -1: print("gubcovers  ", end='')
        print()
        
        print(f"--> Cut passes : {model.parameters.mip.limits.cutpasses} \t\t\t(>0 number of cut passes; 0 automatic (default); -1 None)")
        print(f"--> Presolve: {model.parameters.preprocessing.presolve} \t\t\t\t(1 yes; 0 no)")
        print(f"--> Presolve node: {model.parameters.mip.strategy.presolvenode} \t\t(0 automatic (default), -1 no)")
              #f"(-1 no; "
              #f"0 automatic (default))") # , "
              # f"1 force presolve at nodes, "
              # f"2 probing on integer-infeasible vars,"
              # f"3 aggressive node probing.")
        print(f"--> Reduce: {model.parameters.preprocessing.reduce} \t\t\t\t(0 no primal or dual reductions)")
        # print(f"--> Aggregator {model.parameters.preprocessing.aggregator}")
        # print(f"--> Search strategy: {model.parameters.mip.strategy.search}")

    def set_fast_branching(self, n, x):
        # Now, prepare lists for variables, weights, and branch directions.
        # For each (i,j) pair, we want the variable with k=2 to have the highest weight, k=3 next, etc.
        ordered_vars = []
        ordered_weights = []
        ordered_brdirs = []  # We'll use 0 (no preferred branch direction) for all.

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                # For the k-index values in increasing order.
                for k in range(2, n):
                    # Append the variable.
                    ordered_vars.append(x[(i, j, k)])
                    # Weight is chosen such that lower k gets a higher value.
                    # For example, when k=2, weight = n-2; when k=3, weight = n-3, etc.
                    ordered_weights.append(n - k)
                    # No specific branch direction is set (0 means no preference).
                    ordered_brdirs.append(0)

        # Set the ordering on these variables.
        self.set_ordering(ordered_vars, ordered_weights, ordered_brdirs)


    def set_ordering(self, dvars, weights, brdirs):
        """
        Set a custom branching ordering for a list of decision variables.

        Parameters:
          dvars   : list of docplex decision variables
          weights : list of numerical weights (higher means higher priority)
          brdirs  : list of branch directions (-1, 0, or 1); 0 means no preference
        """
        ldvars = list(dvars)
        lweights = list(weights)
        ldirs = list(brdirs)
        if ldvars:
            # Get the model from the first variable in the list.
            m = ldvars[0].model
            cpx = m.get_cplex()
            # Build the ordering list using each variable's index.
            ordering_list = [(dv.index, w, brd) for dv, w, brd in zip(ldvars, lweights, ldirs)]
            cpx.order.set(ordering_list)
            # Optionally, write the ordering file (for diagnostic purposes)
            # cpx.order.write('%s_prio.ord' % m.name)


