from enum import Enum
from os.path import join
import inspect

from util.constants import DefaultValues
from instance import instance_reader
from util import functions

class SolverOptions:
    def __init__(self):
        self.default = True
        # TODO: convert to Pyomo
        #  self.displayConflictRefinement = True


class SolverObjectiveType:
    class ObjType(Enum):
        default = "Default"

    def __init__(self):
        pass

    @classmethod
    def add_objective(cls, enum):
        merged = {}
        for x in cls.ObjType:
            merged[x.name] = x.value
        for x in enum:
            merged[x.name] = x.value
        cls.ObjType = Enum(cls.ObjType.__name__, merged)

    def is_valid_objective(self, value):
        return any(x.value == value for x in self.ObjType)


class SolverIO:
    def __init__(self, model_name=DefaultValues.string, lp_out_path=DefaultValues.string, txt_out_path=DefaultValues.string,
                 summary_out_path=DefaultValues.string, instance_path=DefaultValues.string):
        self.model_name = model_name
        self.lp_out_path = lp_out_path
        self.txt_out_path = txt_out_path
        self.summary_out_path = summary_out_path
        self.instance_path = instance_path

    @classmethod
    def fromSolverIO(cls, solverIO):
        return cls(solverIO.instance, solverIO.lp_out_path, solverIO.txt_out_path,
                   solverIO.summary_out_path, solverIO.instance_path)

    def set_input_path(self, instance_path):
        self.instance_path = instance_path

    def set_output_paths(self, lp_out_path, txt_out_path, summary_out_path):
        self.lp_out_path = lp_out_path
        self.txt_out_path = txt_out_path
        self.summary_out_path = summary_out_path


class Solver:
    '''
    TODO: convert to Pyomo
    from docplex.util.environment import get_environment
    from docplex.mp.conflict_refiner import ConflictRefiner
    from docplex.mp.relaxer import Relaxer

    def refine_conflicts(self):
        cr = ConflictRefiner()
        crr = cr.refine_conflict(self.model, display=True)

    def relax(self):
        rx = Relaxer()
        rs = rx.relax(self.model)
        rx.print_information()
        rs.display()
    '''

    def __init__(self, time_limit=None, mipgap=0.0001):
        self.solverOptions = SolverOptions()
        self.solverObjectiveType = SolverObjectiveType()
        self.solverIO = SolverIO()
        self.time_limit = time_limit
        self.mipgap = mipgap

    def build_model(self):
        return None

    @staticmethod
    def read_instance(instance_path, filename):
        return instance_reader.read_instance(join(instance_path, filename))

    def set_objective(self, model, rule, sense=1):
        pass

    def solve(self, model=None, backend="cplex_direct", log_output=False, solver_options=None):
        pass

    def solve_relaxed(self, model, backend="cplex_legacy", log_output=False, solver_options=None):
        pass