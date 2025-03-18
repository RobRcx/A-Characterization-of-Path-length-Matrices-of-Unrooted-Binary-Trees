import os
from src.util import constants, functions
from src.instance import instance_reader

output_path = "out/satisfied_buneman"

if __name__ == "__main__":
    solution_folder_path = os.path.join("..", constants.BMEP_original_output_path)
    for filename in os.listdir(solution_folder_path):
        n, tau = instance_reader.read_instance(os.path.join(solution_folder_path, filename))

        splits = filename.split('_')
        new_filename = splits[0] + "_" + splits[1] + ".txt"
        fout = open(os.path.join(output_path, f"{new_filename}"), "w+")

        for i in range(n):
            if i > 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                for p in range(n):
                    if p == i or p == j:
                        continue
                    for q in range(n):
                        if q == i or q == j or p == q:
                            continue
                        idx = -1
                        if tau[i][j] + tau[p][q] + 2 <= tau[i][q] + tau[j][p] == tau[i][p] + tau[j][q]:
                            idx = 1
                        elif tau[i][q] + tau[j][p] + 2 <= tau[i][j] + tau[p][q] == tau[i][p] + tau[j][q]:
                            idx = 2
                        elif tau[i][p] + tau[j][q] + 2 <= tau[i][j] + tau[p][q] == tau[i][q] + tau[j][p]:
                            idx = 3
                        fout.write(f"{idx} {i + 1} {j + 1} {p + 1} {q + 1}\n")
        fout.close()
