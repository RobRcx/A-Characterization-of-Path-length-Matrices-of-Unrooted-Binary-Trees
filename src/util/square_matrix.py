"""
Classes:
    SquareMatrix
        Manages a square matrix with basic properties checking such as symmetry, diagonal, and identity.
"""

import numpy as np
from itertools import combinations
import copy

class SquareMatrix:
    """
    Manages a square matrix for basic mathematical and property-checking operations.

    This class initializes a square matrix of a given size and allows the user to load, analyze, and print the matrix.
    It provides methods to check if the matrix is symmetric, diagonal, or an identity matrix.

    Attributes:
        size (int): The size of the square matrix.
        matrix (np.ndarray): The numpy array representing the square matrix.
        modified_matrix (np.ndarray): The numpy array representing the square matrix obtained by applying method apply_operation_to_elements.
        power_matrix (np.ndarray): The numpy array representing the square matrix obtained by applying method apply_operation_to_elements.
    """

    def __init__(self, size=None):
        """
        Initializes a new SquareMatrix instance with a specified size, if size is different from None.

        Args:
            size (int): The size of the square matrix, defining both the number of rows and columns.
        """
        self.size = size
        if size is not None:
            self.matrix = np.zeros((size, size))
        else:
            self.matrix = None
        self.modified_matrix = None
        self.power_matrix = None
        if size is not None:
            self.contracted_indices = [False for _ in range(self.size)]
        else:
            self.contracted_indices = None

    def read_matrix(self, input_matrix):
        """
        Reads a matrix from the input and assigns it to the matrix attribute.

        This method sets the matrix attribute to a numpy array of the input matrix if it has the correct dimensions.

        Args:
            input_matrix (list[list[float]] | np.ndarray): A 2D list or numpy array representing the square matrix.

        Raises:
            ValueError: If the input matrix does not match the required size.
        """
        if np.array(input_matrix).shape == (self.size, self.size):
            self.matrix = np.array(input_matrix)
            self.apply_power_of_two()
        else:
            raise ValueError("Input matrix must be of size {}x{}".format(self.size, self.size))

    def read_matrix_from_file(self, file_path):
        """
        Reads a square matrix from a specified text file and adjusts the matrix size based on the file content.

        This method reads a matrix from a text file where each line corresponds to a row in the matrix and
        each value in the line is separated by spaces. It dynamically adjusts the size of the matrix based
        on the number of rows and the number of columns in the first row, assuming the matrix is square.

        Args:
            file_path (str): The file path from which to read the matrix.

        Raises:
            ValueError: If the matrix read from the file is not square.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
            matrix = [list(map(float, line.strip().split())) for line in lines]
            if any(len(row) != len(matrix) for row in matrix):
                raise ValueError("The matrix must be square (same number of rows and columns)")
            self.size = len(matrix)
            self.matrix = np.array(matrix)
            self.apply_power_of_two()
            if self.contracted_indices is None:
                self.contracted_indices = [False for _ in range(self.size)]

    def print_matrix_statistics(self):
        """
        Prints the maximum element,  the minimum element strictly greater than zero and other statistics of  the matrix.

        This method finds the maximum value in the matrix and the minimum value that is strictly greater than zero.
        In addition it computes othe statistics..
        It assumes the matrix has been loaded and contains valid numeric entries.

        Raises:
            ValueError: If no matrix has been loaded prior to invoking this method.
        """
        if self.matrix is None:
            raise ValueError("No matrix has been loaded.")

        # Find the maximum element in the matrix
        max_element = np.max(self.matrix)

        # Find the minimum element strictly greater than zero
        min_above_zero = np.min(self.matrix[self.matrix > 0]) if np.any(self.matrix > 0) else None

        # Print the results
        print(f"Maximum element in the matrix: {max_element}")
        if min_above_zero is not None:
            print(f"Minimum element in the matrix strictly greater than 0: {min_above_zero}")
        else:
            print("There are no elements in the matrix strictly greater than 0.")

        print(f"expected number of internal nodes: {int(self.size - 2)}")

        print(f"expected number of edges: {int(2 * self.size - 3)}")

    def is_symmetric(self):
        """
        Checks if the matrix is symmetric.

        This method compares the matrix with its transpose and returns True if they are identical, otherwise False.

        Returns:
            bool: True if the matrix is symmetric, False otherwise.
        """
        return np.allclose(self.matrix, self.matrix.T)

    def is_diagonal(self):
        """
        Checks if the matrix is diagonal.

        This method verifies if all the off-diagonal elements of the matrix are zero.

        Returns:
            bool: True if the matrix is diagonal, False otherwise.
        """
        return np.allclose(self.matrix, np.diag(np.diagonal(self.matrix)))

    def is_identity(self):
        """
        Checks if the matrix is an identity matrix.

        This method compares the matrix to an identity matrix of the same size.

        Returns:
            bool: True if the matrix is an identity matrix, False otherwise.
        """
        return np.allclose(self.matrix, np.eye(self.size))

    def apply_operation_to_elements(self, func):
        """
        Applies a specified function to each element of the matrix and stores the result in a new attribute.

        This method applies a given function to every element of the original matrix 'matrix' and saves the result
        in 'modified_matrix'. The function should take a single number and return a single number.

        Args:
            func (function): A function that takes a single float and returns a float. This function will be
                             applied to each element of the matrix.

        Raises:
            ValueError: If no matrix has been loaded prior to the operation.
        """
        if self.matrix is None:
            raise ValueError("No matrix has been loaded.")

        self.modified_matrix = np.vectorize(func)(self.matrix)

    def apply_power_of_two(self):
        """
        Applies the power of 2 to each element of the matrix and stores the result in a new attribute.

        This method applies the power of 2 function to every element of the original matrix 'matrix' and saves the result
        in 'power_matrix'. The function should take a single number and return a single number.

        Args:

        Raises:
            ValueError: If no matrix has been loaded prior to the operation.
        """
        if self.matrix is None:
            raise ValueError("No matrix has been loaded.")

        # Definizione della funzione da applicare
        def power_of_two(x): return 2 ** (-x)

        # Applicazione della funzione
        self.power_matrix = np.vectorize(power_of_two)(self.matrix)

    def sum_rows(self, matrix_to_check):
        """
        Returns a vector where each element is the sum of the elements of each row in the matrix_to_check.

        This method calculates the sum of elements for each row of the matrix_to_check and returns a numpy array
        containing these sums. It assumes the matrix has been loaded and is not empty.

        Args:
            matrix_to_check (np.ndarray): Matrix of which the elements of the rows are to be summed

        Returns:
            np.ndarray: A 1D numpy array where each element is the sum of the elements of the corresponding row in the matrix_to_check.

        Raises:
            ValueError: If no matrix has been loaded prior to invoking this method.
        """
        if matrix_to_check is None:
            raise ValueError("No matrix has been loaded.")

        return np.sum(matrix_to_check, axis=1)

    def check_row_sums(self, matrix_to_check=None, row_sum=1.5):
        """
        Checks if the sum of elements in each row of the matrix_to_check equals row_sum. If not, returns the indices
        of the rows that do not meet this criterion along with their sums.

        This method uses the `sum_rows` method to compute the sum of elements in each row, then checks
        if each sum equals row_sum. It collects and returns any rows that do not meet this condition.

        Args:
            matrix_to_check (np.ndarray): Matrix whose row properties should be checked.
            row_sum (float): Expected value of the sum of the elements of the row of the matrix.

        Returns:
            list[tuple[int, float]]: A list of tuples, each containing the index of the row and the sum
                                      of its elements, for rows where the sum does not equal row_sum.

        Raises:
            ValueError: If no matrix has been loaded prior to invoking this method.
        """
        matrix_to_check = self.power_matrix if matrix_to_check is None else matrix_to_check

        if matrix_to_check is None:
            raise ValueError("No matrix has been loaded.")

        row_sums = self.sum_rows(matrix_to_check)
        mismatches = [(index, sum_value) for index, sum_value in enumerate(row_sums) if
                      not np.isclose(sum_value, row_sum)]
        return True if len(mismatches) == 0 else mismatches

    def check_manifold(self, target_value=None):
        """
        Flattens the 'matrix' and 'power_matrix', computes their scalar product, and checks
        if it equals a specified input value.

        This method first flattens both the 'matrix' and the 'power_matrix', then computes the scalar
        product of these flattened arrays. It compares the result of the scalar product with the
        'target_value' provided as an argument and returns True if they are equal, False otherwise.

        Args:
            target_value (float): The value to compare against the scalar product of the flattened matrices.

        Returns:
            bool: True if the scalar product equals the 'target_value', False otherwise.

        Raises:
            ValueError: If either 'matrix' or 'modified_matrix' has not been loaded or defined.
        """
        if self.matrix is None or self.power_matrix is None:
            raise ValueError("Both 'matrix' and 'power_matrix' must be loaded/defined.")

        target_value = 2 * self.size - 3 if target_value is None else target_value

        # Flatten the matrices
        flat_matrix = self.matrix.flatten()
        flat_power_matrix = self.power_matrix.flatten()

        # Compute the scalar product
        scalar_product = np.dot(flat_matrix, flat_power_matrix)

        # Check if the scalar product is equal to the target value
        ret = np.isclose(scalar_product, target_value)
        return ret if ret else scalar_product

    def verify_nplets_condition(self, n, condition_func):
        """
        Verifies a specified condition for each n-plet of distinct row indices and prints the
        n-plets that do not satisfy the condition.

        This method iterates through all possible combinations of n distinct rows and
        applies a user-defined function 'condition_func' to these rows. If the condition is not
        satisfied, the method prints the indices of these rows.

        Args:
            condition_func (function): A function that takes the n-plets (as list) and
                                       returns a boolean indicating whether the specified condition
                                       is met and the value of the function

        Raises:
            ValueError: If no matrix has been loaded or if the matrix has fewer than three rows.
        """
        if self.matrix is None:
            raise ValueError("No matrix has been loaded.")
        if self.size < n:
            raise ValueError("Matrix must have at least n rows to form distinct triplets.")

        row_indices = range(self.size)
        failed_n_plets = []

        # Iterate over each combination of three distinct rows
        for n_plet in combinations(row_indices, n):
            ret, val = condition_func(list(n_plet))
            if not ret: failed_n_plets.append((n_plet, val))

        # Print failed triplets
        if failed_n_plets:
            print("N-plets that do not satisfy the condition:")
            print(*failed_n_plets, sep='\n')
        else:
            print("All n-plets satisfy the condition.")

    def print_matrix(self):
        """
        Prints the matrix.

        This method displays the current state of the matrix attribute to the standard output.
        """
        print(self.matrix)

    def strong_triangular_inequality(self, pair):
        """
        Checks the strong triangular inequality condition for a given pair or triplet of indices in the matrix.

        This method evaluates the strong triangular inequality for a specified pair opr triplet of
        indices. The condition checked is whether the sum of the distances from the first element to each of the
        two last elements in the pair/triplets minus the distance between the two elements in the pair is at least 2 and even.
        The method assumes that the distance matrix is already set and the indices are within the correct range.

        Args:
            pair (tuple): A tuple of two ot three integers representing the indices in the matrix to check the inequality
                          condition between. In case of two indices the third index is set to 0 by default.

        Returns:
            bool: True if the strong triangular inequality condition is met, False otherwise. If either index
                  in the pair is 0, the function returns True as a boundary condition.
            diff: value of the differences of the length of the distances involved in the triangular inequality.

        Examples of usage:
            - To check the condition between indices 1 and 2, call:
              result, diff  = matrix.strong_triangular_inequality((1, 2))
              This will return True if the condition is met.
        """
        if len(pair) == 2:
            i, j, k = 0, pair[0], pair[1]
            if j == 0 or k == 0:  # Boundary condition handling
                return True, 0
        else:
            i, j, k = pair[0], pair[1], pair[2]

        # Compute the condition value
        diff = self.matrix[i, k] + self.matrix[j, k] - self.matrix[i, j]

        # Check if the difference is at least 2 and is even
        return (diff >= 2), diff  # return (diff >= 2) and ((diff % 2) == 0), diff

    def strong_buneman_inequality(self, triplet):
        """
        Checks the strong Buneman inequality conditions for a given triplet or quadruplet of indices in the matrix.

        This method evaluates the strong Buneman inequalities on the first row of the matrix for a specified triplet or quadruplet of
        indices. It checks whether the distance between the four cosidered elements is at least 2 and even.
        The method assumes that the distance matrix is already set and the indices are within the correct range.

        Args:
            pair (tuple): A tuple of three ot four integers representing the indices in the matrix to check the inequality
                          condition between. In case of three indices the fourth index is set to 0 by default.

        Returns:
            bool: True if the Buneman inequality conditions are met, False otherwise. If either index
                  in the triplet is 0, the function returns True as a boundary condition.
            quad: value of the differences of the length of the distances involved in the Buneman inequalities and the
                  a quadruplet indicating the order in which the indices are considered.
        """
        if len(triplet) == 3:
            i, j, p, q = 0, triplet[0], triplet[1], triplet[2]
            if p == 0 or q == 0 or j == 0:  # Boundary condition handling
                return True, 0
        else:
            i, j, p, q = triplet[0], triplet[1], triplet[2], triplet[3]

        # Compute the condition value
        if (self.matrix[i, p] + self.matrix[j, q]) == (self.matrix[i, q] + self.matrix[j, p]):
            diff = self.matrix[i, p] + self.matrix[j, q] - (self.matrix[i, j] + self.matrix[p, q])
            quad = (i, j, p, q)
        elif (self.matrix[i, j] + self.matrix[p, q]) == (self.matrix[i, q] + self.matrix[j, p]):
            diff = self.matrix[i, j] + self.matrix[p, q] - (self.matrix[i, p] + self.matrix[j, q])
            quad = (i, p, j, q)
        elif (self.matrix[i, j] + self.matrix[p, q]) == (self.matrix[i, p] + self.matrix[j, q]):
            diff = self.matrix[i, j] + self.matrix[p, q] - (self.matrix[i, q] + self.matrix[j, p])
            quad = (i, q, j, p)
        else:
            diff = 0
            quad = (0, 0, 0, 0)

        # Check if the difference is at least 2 and is even
        return (diff >= 2) and ((diff % 2) == 0), (diff, quad)

    def find_entry_equal_to(self, val):
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] == 2:
                    return i, j
        return -1, -1

    def contract(self, i, j):
        n = len(self.matrix)

        # Zero the i-th row and column
        for k in range(n):
            if k != i:
                self.matrix[i][k] = 0
                self.matrix[k][i] = 0

        # Subtract 1 from the j-th row and column except the diagonal
        for k in range(n):
            if k != j and k != i and self.contracted_indices[k] is False:
                self.matrix[j][k] -= 1
                self.matrix[k][j] -= 1

        self.contracted_indices[i] = True

    def permute(self, i, j):
        i -= 1; j -= 1
        n = len(self.matrix)
        tmp = copy.deepcopy(self.matrix)
        for k in range(n):
            if k == i or k == j:
                continue
            tmp[i][k] = self.matrix[j][k]
            tmp[j][k] = self.matrix[i][k]
            tmp[k][i] = self.matrix[k][j]
            tmp[k][j] = self.matrix[k][i]
        self.matrix = tmp

    def clean_print(self):
        n = len(self.matrix)
        for i in range(n):
            for j in range(n):
                print(f"{int(self.matrix[i][j])}", end=" ")
            print()