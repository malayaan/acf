import numpy as np
import matplotlib.pyplot as plt

def compute_burt_matrix(profiles):
    """Compute the Burt matrix."""
    n, m = profiles.shape
    outer_product = np.einsum('ij,ik->ijk', profiles, profiles)
    denom = np.einsum('i,j->ij', profiles.sum(axis=1), profiles.sum(axis=0))
    return (outer_product / denom).sum(axis=0)

def center_matrix(matrix, axis):
    """Center a matrix along the given axis."""
    return matrix - matrix.mean(axis=axis, keepdims=True)

def project_on_eigenvectors(matrix, eigenvectors):
    """Project the matrix on the provided eigenvectors."""
    return np.dot(matrix, eigenvectors)

# Read data from file
with open('sondage_parfait.txt', 'r') as file:
    lines = file.readlines()[1:]  # skip header
    data = [line.strip().split(', ') for line in lines]

# Extract unique values for the first and second columns and sort them
unique_values_1 = sorted(list(set([pair[0] for pair in data])))
unique_values_2 = sorted(list(set([pair[1] for pair in data])))

# Build contingency table
contingency_table = np.zeros((len(unique_values_1), len(unique_values_2)))

for pair in data:
    i = unique_values_1.index(pair[0])
    j = unique_values_2.index(pair[1])
    contingency_table[i, j] += 1

# Compute profiles
profiles = contingency_table / contingency_table.sum()

# Compute Burt matrix
burt_matrix = compute_burt_matrix(profiles)

# Center Burt matrix along rows and columns
centered_burt_rows = center_matrix(burt_matrix, axis=1)
centered_burt_columns = center_matrix(burt_matrix, axis=0)
print(centered_burt_columns)
# Compute eigenvalues and eigenvectors for row-centered matrix
eigenvalues, eigenvectors = np.linalg.eig(centered_burt_rows)
print(eigenvalues)
# Sort by descending eigenvalue
eigenorder = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, eigenorder]

# Project data onto first two eigenvectors
scores_rows = project_on_eigenvectors(centered_burt_rows, eigenvectors[:, :2])

# Compute eigenvalues and eigenvectors for columns-centered matrix
eigenvalues, eigenvectors = np.linalg.eig(centered_burt_columns)

# Sort by descending eigenvalue
eigenorder = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, eigenorder]

scores_columns = project_on_eigenvectors(centered_burt_columns, eigenvectors[:, :2])

# Visualization
labels_rows = unique_values_1
labels_columns = unique_values_2

plt.figure(figsize=(15, 12))

for j, label in enumerate(labels_rows):
    plt.scatter(scores_rows[j, 0], scores_rows[j, 1], label=f"{label} (rows)", 
                s=150, marker='o', edgecolors='r', facecolors='none')
    plt.scatter(scores_columns[j, 0], scores_columns[j, 1], label=f"{labels_columns[j]} (columns)", 
                s=100, marker='x', color='b')
    plt.annotate(label, (scores_rows[j, 0], scores_rows[j, 1]), fontsize=12)
    plt.annotate(labels_columns[j], (scores_columns[j, 0], scores_columns[j, 1]), fontsize=12, alpha=0.7)

plt.axhline(0, color='grey', linewidth=0.5)
plt.axvline(0, color='grey', linewidth=0.5)
plt.title('ACM avec Matrices Centrées sur Lignes et Colonnes')
plt.xlabel('Première dimension')
plt.ylabel('Deuxième dimension')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
