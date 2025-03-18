import os
import time
import datetime

from util import circular_order_verifier
from util import constants
from util.constants import BMEPMode, SolverMode, RunOptions
from util import config
from instance import instance_reader

def solve(n, m, conf, filename, instance):
    # If n is not within the bounds, continue
    if not check_n(n, conf, instance):
        print(f"Instance {instance}: Number of taxa equal to {n}: greater than {conf.n_max}"
              f" or less than {conf.n_min}. Skipping.")
        return True, None, None, None, None, None, None, None, None, None

    # Read custom buneman and circular orders constraints
    read_buneman_custom_constraints(filename, conf)
    read_circular_orders_custom_constraints(n, conf)

    print(f"Instance \t= {instance}\n"
          f"n \t\t= {n}\n"
          f"Objective \t= {conf.obj_type}")

    solverIO = conf.solver_io_cls(instance=instance, obj_type=conf.obj_type, config_name=conf.name)

    if is_solved(instance, solverIO):
        print(f"Instance {instance} already solved.")
        return True, False, None, None, None, None, None, None, None, None

    # print("Execution parameters : ", params)

    conf.params.update({"model_name": instance, "n": n, "m": m, "obj_type": conf.obj_type, "solverIO": solverIO})

    print(
        f"Printing solution to {solverIO.txt_out_path}\n"
        f"Printing summary to {solverIO.summary_out_path}"
    )

    '''
        Start integer solving with timing.
    '''
    solver = conf.solver_cls(**conf.params)

    # print(f"{repeat} entries set to be equal to 2 in upper triangular matrix")
    # solver.build_model(repeat)

    solver.build_model()
    solver.select_solver_mode(solver, conf.solver_mode)
    solver.print_solver_info(model=solver.model)

    # Solve MIP
    print(f"Solving integral version of the instance...\nStart: {datetime.datetime.now()}.")
    start = time.time()
    # solver.model.pprint()
    ok, results, res_map = solver.solve(model=solver.model, log_output=RunOptions.debug)
    end = time.time()
    print(f"End: {datetime.datetime.now()}.")
    if not ok:
        print(f"Could not solve MIP formulation of instance {instance}.")
        return False, False, None, None, False, None, None, None, solver, solverIO
    res_map["time"] = end - start

    # Solve relaxed
    print(f"Solving relaxed version of the instance...\nStart: {datetime.datetime.now()}.")
    start = time.time()
    ok, relaxed_results, relaxed_res_map = solver.solve_relaxed(model=solver.model, log_output=RunOptions.debug)
    end = time.time()
    print(f"End: {datetime.datetime.now()}.")
    if not ok:
        print(f"Could not solve relaxed formulation of instance {instance}.")
        return False, True, results, res_map, False, None, None, None, solver, solverIO
    relaxed_res_map["time"] = end - start

    gap = 100 * (res_map["obj"] - relaxed_res_map["obj"]) / res_map["obj"] if res_map["obj"] != 0 else -1.0
    return False, True, results, res_map, True, relaxed_results, relaxed_res_map, gap, solver, solverIO


def main(conf):
    for filename in os.listdir(conf.instance_path):
        # Checks that the current file is not a directory
        if os.path.isdir(os.path.join(conf.instance_path, filename)):
            continue

        # Gets instance name from filename and reads instance
        instance = filename.split(".")[0]
        n, m = conf.solver_cls.read_instance(conf.instance_path, filename)

        skipped, ok_mip, results, res_map, ok_relaxed, relaxed_results, relaxed_res_map, gap, solver, solverIO = (
            solve(n, m, conf, filename, instance))

        if skipped:
            continue

        print_results(results, res_map, relaxed_results, relaxed_res_map, gap)

        if RunOptions.file_export:
            export_solution(instance, res_map, relaxed_res_map, gap, solver, solverIO)
        print(f"Solution written to {solverIO.txt_out_path}.")


def main_buneman_violation(conf):
    for n in range(conf.n_min, conf.n_max + 1):
        for i in range(conf.start, conf.start + conf.repeat):
            m = instance_reader.generate(n)
            instance = f"{i}_n{n}"
            skipped, ok_mip, results, res_map, ok_relaxed, relaxed_results, relaxed_res_map, gap, solver, solverIO = (
                solve(n, m, conf, "", instance))

            if skipped:
                continue

            if ok_mip:
                print_results(results, res_map, relaxed_results, relaxed_res_map, gap)

                solverIO.txt_out_path = os.path.join(solverIO.output_folder_path, f"{i}_n{n}_sol.txt")
                if RunOptions.file_export:
                    export_solution(instance, res_map, relaxed_res_map, gap, solver, solverIO)
                print(f"Solution written to {solverIO.txt_out_path}.")

                # Compute violated circular orders and store them in a file
                count = circular_order_verifier.main(solverIO.txt_out_path)
                print(f"{count} violated circular orders written to {solverIO.txt_out_path}.")

            # Hardcoded exit
            if conf.entries_equal_to_two is not None and not ok_mip:
                conf.params["entries_equal_to_two"] -= 1
                if conf.params["entries_equal_to_two"] < 2:
                    exit(-2)

# Checks whether n is within the bounds
def check_n(n, conf, instance):
    if n > conf.n_max or n < conf.n_min:
        return False
    return True

def is_solved(instance, solverIO):
    # Get solved instances in summary
    solved_instances = set()
    lines = []
    if os.path.exists(solverIO.summary_out_path):
        with open(solverIO.summary_out_path, "r+") as f:
            lines = f.readlines()
            for x in lines:
                splits = x.split(' ')
                if splits[0] != "instance":
                    solved_instances.add(splits[0])
            f.close()
    # Add header of file export (if required)
    if len(lines) == 0:
        with open(solverIO.summary_out_path, "w+") as f:
            f.write(f"instance obj time nodes cuts rlx time gap\n")
    # Check solved instances in individual result files or summary
    if RunOptions.skip_solved and os.path.exists(solverIO.txt_out_path) and (instance in solved_instances):
        print(f"Instance {instance} already solved. Skipping.")
        return True
    return False

def read_buneman_custom_constraints(filename, conf):
    if conf.buneman_custom_constraints:
        valid_buneman_path = os.path.join(conf.instance_path, constants.Path.buneman_folder_name)
        buneman_quadruplets = [[] for _ in range(3)]
        buneman_file = os.path.join(valid_buneman_path, filename)
        if os.path.exists(buneman_file):
            with open(buneman_file) as f:
                lines = f.readlines()
                for l in lines:
                    which, i, j, p, q = (int(x) for x in l.split())
                    buneman_quadruplets[which - 1].append([i, j, p, q])
        else:
            print(f"Custom Buneman constraints options activated, but file {buneman_file} not found!")
            print(f"   Current working directory: {os.getcwd()}")
        conf.params.update({"buneman_quadruplets": buneman_quadruplets})

def read_circular_orders_custom_constraints(n, conf):
    if conf.circular_orders_custom_constraints:
        if os.path.exists(conf.circular_orders_custom_filepath):
            with open(conf.circular_orders_custom_filepath, "r") as f:
                lines = f.readlines()
                circular_orders = [[0 for _ in range(n)] for _ in range(len(lines))]
                for i in range(len(lines)):
                    splits = lines[i].split(' ')
                    for j in range(n):
                        circular_orders[i][j] = int(splits[j])
                conf.params.update({"circular_orders": circular_orders})
        else:
            print(f"Custom circular orders file {conf.circular_orders_custom_filepath} not found!")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Current working directory content: {os.listdir(os.getcwd())}")


def export_solution(instance, res_map, relaxed_res_map, gap, solver, solverIO):
    f = open(solverIO.summary_out_path, "a")
    # instance obj time nodes cuts rlx time nodes gap
    f.write(f"{instance} {res_map['obj']} {res_map['time']} {res_map['nodes']} {res_map['cuts']} "
            f"{relaxed_res_map['obj']} {relaxed_res_map['time']} {gap}\n")
    f.close()
    solver.export_solution(solverIO.txt_out_path)

def print_results(results, res_map, relaxed_results, relaxed_res_map, gap):
    if RunOptions.debug:
        print("Integral solution:\n", results)
    if RunOptions.debug:
        print("Integral solution:\n", relaxed_results)
    print("***** Summary *****")
    print(f"* time = {res_map['time']} (sec)")
    print(f"* obj = {res_map['obj']}")
    print(f"* rlx = {relaxed_res_map['obj']}")
    print(f"* gap = {gap:.4}%")
    print("*******************")

if __name__ == "__main__":
    '''configs = [config.Config(mode=BMEPMode.BunV, init_file_path="ini/config_Buneman_violation_n12.ini"),
               config.Config(mode=BMEPMode.BunV, init_file_path="ini/config_Buneman_violation_n13.ini"),
               config.Config(mode=BMEPMode.BMEPbun, init_file_path="ini/config_F1_Manifold.ini"),
               config.Config(mode=BMEPMode.BMEPbun, init_file_path="ini/config_F1_Buneman.ini"),
               config.Config(mode=BMEPMode.BMEPbun, init_file_path="ini/config_contractions.ini"),
               config.Config(mode=BMEPMode.BMEPbun, init_file_path="ini/config_contractions_Manifold.ini")]'''
    configs = [config.Config(mode=BMEPMode.BunV, init_file_path="ini/main.ini")]
    for conf in configs:
        print("Solver cls \t= ", conf.solver_cls) #, "\nTime limit : ", conf.time_limit, "\nMip gap : ", conf.mipgap)
        if conf.mode == BMEPMode.BunV:
            print("Mode \t\t= Buneman violation.")
            main_buneman_violation(conf)
        elif conf.mode == BMEPMode.BMEPbun:
            print("Mode \t\t= Formulation 1.")
            main(conf)
        elif conf.mode == BMEPMode.BMEPcon:
            print("Mode \t\t= Formulation based on contractions.")
            main(conf)
        else:
            print("Mode unrecognized. Terminating.")
