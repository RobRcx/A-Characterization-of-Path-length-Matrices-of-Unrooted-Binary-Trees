import numpy as np

# Define the matrix T
T = np.array([
    [0, 2, 3, 4, 5, 6, 7, 7],
    [2, 0, 3, 4, 5, 6, 7, 7],
    [3, 3, 0, 4, 3, 5, 6, 6],
    [4, 4, 4, 0, 3, 4, 4, 4],
    [5, 5, 3, 3, 0, 3, 5, 5],
    [6, 6, 5, 4, 3, 0, 3, 3],
    [7, 7, 6, 4, 5, 3, 0, 2],
    [7, 7, 6, 4, 5, 3, 2, 0]
])

# Check the inequality for all distinct i, j, p
n = T.shape[0]
all_satisfy = True
violations = []

for i in range(n):
    for j in range(n):
        for p in range(n):
            if i != j and i != p and j != p:
                if T[i, j] + T[i, p] - T[j, p] < 2:
                    all_satisfy = False
                    violations.append((i, j, p))

print(all_satisfy)
print(violations)