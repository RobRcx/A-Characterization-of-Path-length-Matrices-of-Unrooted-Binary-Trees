from square_matrix import SquareMatrix

matrix_path = "../../data/output/taxa_permutator_matrix.txt"

# Creazione dell'oggetto
matrix = SquareMatrix()

# Lettura della matrice da file
matrix.read_matrix_from_file(matrix_path)

matrix.print_matrix()

# Custom code for the counterexample matrix
print(f"Permuting 1 and 11")
matrix.permute(1, 11)
matrix.print_matrix()

print(f"Permuting 2 and 3")
matrix.permute(2, 3)
matrix.print_matrix()

print(f"Permuting 6 and 10")
matrix.permute(6, 10)
matrix.print_matrix()

print(f"Permuting 7 and 11")
matrix.permute(7, 11)
matrix.print_matrix()

print(f"Permuting 8 and 12")
matrix.permute(8, 12)
matrix.print_matrix()

# Clean print for file export
matrix.clean_print()


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

print("!!! New matrix !!!")

matrix_path = "../../data/solutions/new/zero/Buneman-Violation/n12/23_n12_sol.txt"
# Creazione dell'oggetto
matrix = SquareMatrix()

# Lettura della matrice da file
matrix.read_matrix_from_file(matrix_path)

matrix.permute(4, 6)
matrix.permute(6, 8)
matrix.permute(10, 11)
matrix.clean_print()
