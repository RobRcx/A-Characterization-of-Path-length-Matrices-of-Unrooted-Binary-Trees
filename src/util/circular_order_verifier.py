import os

from util.functions import (next_permutation)
from instance import instance_reader

def find_violated_circular_orders(n, tau):
    count = 0
    res = []
    for i in range(n):
        print(f"Now computing circular orders excluding {i}. Found {count} violated circular orders.")
        seq = [j for j in range(n) if j != i]
        first = seq[0]
        seq.pop(0)
        ok = True
        while ok:
            lhs = tau[first][seq[0]]
            for j in range(n - 3):
                lhs += tau[seq[j]][seq[j + 1]]
            lhs += tau[seq[n - 3]][first]
            if lhs < 4 * n - 8:
                count += 1
                cur = [first + 1]
                for j in range(n - 2):
                    cur.append(seq[j] + 1)
                cur.append(first + 1)
                res.append(cur)
            ok = next_permutation(seq)
    # print(f"{count} violated circular orders.")
    return count, res

def main(matrix_path=None):
    if matrix_path is None:
        # matrix_path = "../../data/solutions/new/zero/Buneman-Violation/n12/28_n12_sol.txt"
        # Paper counterexample
        n = 12
        tau = [[0, 2, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7],
                [2, 0, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7],
                [4, 4, 0, 2, 6, 6, 5, 5, 7, 7, 7, 7],
                [4, 4, 2, 0, 6, 6, 5, 5, 7, 7, 7, 7],
                [5, 5, 6, 6, 0, 2, 5, 5, 5, 5, 6, 6],
                [5, 5, 6, 6, 2, 0, 5, 5, 5, 5, 6, 6],
                [6, 6, 5, 5, 5, 5, 0, 2, 6, 6, 5, 5],
                [6, 6, 5, 5, 5, 5, 2, 0, 6, 6, 5, 5],
                [7, 7, 7, 7, 5, 5, 6, 6, 0, 2, 4, 4],
                [7, 7, 7, 7, 5, 5, 6, 6, 2, 0, 4, 4],
                [7, 7, 7, 7, 6, 6, 5, 5, 4, 4, 0, 2],
                [7, 7, 7, 7, 6, 6, 5, 5, 4, 4, 2, 0]
        ]
    else:
        n, tau = instance_reader.read_instance(matrix_path)

    output_path = "./util/out/unsatisfied_circular"
    outfile_path = os.path.join(output_path, f"counterexample_n{n}_violated_circular.txt")

    print(f"Computing violated circular orders of {matrix_path}...")
    count, res = find_violated_circular_orders(n, tau)

    if count > 0:
        with open(outfile_path, "a") as f:
            for seq in res:
                for x in seq:
                    f.write(f"{x} ")
                f.write("\n")
            print(f"Completed. Data written to {outfile_path}")

    return count

if __name__ == "__main__":
    main()
