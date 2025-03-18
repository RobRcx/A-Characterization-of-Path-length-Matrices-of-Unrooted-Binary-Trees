import inspect
import numpy as np
import configparser

from model.cplex import BMEP
from model.cplex import BMEPTopological
from util import constants
from util import functions
from util.functions import convert
from util.constants import SolverMode, BMEPMode

from util.functions import resolve_dotted_path


class Config:
    def __init__(self, mode=None, init_file_path=None):
        self.name = "default"
        self.n_min = 12
        self.n_max = 12
        self.manifold = True
        self.time_limit = None
        self.mipgap = 0.00000001
        self.buneman_violation = True
        self.buneman_disjunctive = False
        self.buneman_custom_constraints = False
        self.circular_orders_custom_constraints = True
        self.circular_orders_custom_filepath = "util/out/unsatisfied_circular/counterexample_n12_violated_circular.txt"
        self.solver_mode = SolverMode.Turbo
        self.instance_path = constants.Path.instance_path
        self.mode = mode if mode is not None else constants.BMEPMode.BMEPbun
        self.order_first_row = True
        self.entries_equal_to_two = 3
        self.fast_branching = True
        self.repeat = 100000
        self.start = 1

        # Read configuration from file if required
        if init_file_path is not None:
            config = self.read_config_from_file(init_file_path)
            self.name = config['Settings']['name'] if 'name' in config['Settings'] else self.name
            self.n_min = convert(int, config['Settings']['n_min'])
            self.n_max = convert(int, config['Settings']['n_max'])
            self.manifold = convert(bool, config['Settings']['manifold'])
            self.time_limit = convert(float, config['Settings']['time_limit'])
            self.mipgap = convert(float, config['Settings']['mipgap'])
            self.buneman_violation = convert(bool, config['Settings']['buneman_violation'])
            self.buneman_disjunctive = convert(bool, config['Settings']['buneman_disjunctive'])
            self.buneman_custom_constraints = convert(bool, config['Settings']['buneman_custom_constraints'])
            self.circular_orders_custom_constraints = convert(bool, config['Settings']['circular_orders_custom_constraints'])
            self.circular_orders_custom_filepath = config['Settings']['circular_orders_custom_filepath'] if self.circular_orders_custom_constraints else None
            self.solver_mode = eval(config['Settings']['solver_mode'])
            self.instance_path = eval(config['Settings']['instance_path']) # resolve_dotted_path(config['Settings']['instance_path'])
            self.mode = eval(config['Settings']['mode'])
            self.order_first_row = convert(bool, config['Settings']['order_first_row'])
            self.entries_equal_to_two = convert(int, config['Settings']['entries_equal_to_two'])
            self.fast_branching = convert(bool, config['Settings']['fast_branching'])
            self.start = convert(int, config['Settings']['start']) if 'start' in config['Settings'] else self.start
            self.repeat = convert(int, config['Settings']['repeat']) if 'repeat' in config['Settings'] else self.repeat


        # Compute parameters map
        self.params = {}
        self.set_params_map()

        # Select execution mode
        self.obj_type = self.solver_cls = self.solver_io_cls = None
        self.select_mode()

        # Set numpy printing options
        np.set_printoptions(linewidth=1000, threshold=np.inf)

        print(self.name)

    def set_params_map(self):
        self.params.update({"manifold": self.manifold,
                       "time_limit": self.time_limit,
                       "mipgap": self.mipgap,
                       "buneman_violation": self.buneman_violation,
                       "buneman_disjunctive": self.buneman_disjunctive,
                       "order_first_row": self.order_first_row,
                       "entries_equal_to_two": self.entries_equal_to_two,
                       "fast_branching": self.fast_branching, })

    def select_mode(self):
        if self.mode.value == constants.BMEPMode.BMEPbun.value:
            self.obj_type = BMEP.ObjType.classic
            self.solver_cls = BMEP.BMEPSolver
            self.solver_io_cls = BMEP.BMEPSolverIO
        elif self.mode.value == constants.BMEPMode.BMEPcon.value:
            self.obj_type = BMEPTopological.ObjType.classic
            self.solver_cls = BMEPTopological.BMEPSolver
            self.solver_io_cls = BMEPTopological.BMEPSolverIO
        elif self.mode.value == constants.BMEPMode.BunV.value:
            self.obj_type = BMEP.ObjType.zero
            self.solver_cls = BMEP.BMEPSolver
            self.solver_io_cls = BMEP.BMEPSolverIO
        else:
            stack = inspect.stack()
            functions.error_exit(message=f"Execution mode {self.mode}: unimplemented or unrecognized.",
                                 code=-1,
                                 invoking_script=inspect.getfile(inspect.currentframe()),
                                 line=stack[-1].lineno)

    def read_config_from_file(self, init_file_path):
        config = configparser.ConfigParser()
        config.read(init_file_path)
        return config