import os
import numpy as np
from enum import Enum

import util.functions as functions

class BMEPMode(Enum):
    BMEPbun = "BMEP buneman"
    BMEPcon = "BMEP contractions"
    Contr = "contractions"
    UBTchk = "UBT check"
    BunV = "Buneman Violation"

class SolverMode(Enum):
    Default = "Default"
    Barebone = "Barebone"
    Turbo = "Turbo"

class Path:
    data_path = "../data"
    solutions_output_path = os.path.join(data_path, "solutions")  # ./data/solutions
    new_solutions_output_path = os.path.join(solutions_output_path, "new")  # ./data/solutions/new
    instance_path = os.path.join(data_path, "instances/original")
    buneman_folder_name = "buneman_quadruplets"
    # buneman_violation_instance_path = os.path.join(data_path, "instances/bunv")
    # Legacy (for "main_cplex_legacy.py")
    BMEP_original_output_path = os.path.join(solutions_output_path, "original")  # BMEP Contraction solutions
    contraction_output_path = os.path.join(new_solutions_output_path, "contractions")  # Contractions solutions

class RunOptions:
    print_solution = True
    print_intermediate_info = False
    file_export = True
    skip_solved = True
    debug = False
    # Legacy (for "main_cplex_legacy.py")
    relax = False

class DefaultValues:
    boolean = False
    string = "Default"

def recognize_mode(mode):
    if not any(x.value == mode.value for x in BMEPMode):
        functions.error_exit("Mode unrecognized", -1)



