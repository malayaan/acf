import numpy as np
import matplotlib.pyplot as plt

# Données : table de contingence
data_matrix_updated = np.array([
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 10]
])

# Calcul des fréquences jointes (f_ij)
total = np.sum(data_matrix_updated)
f_ij = data_matrix_updated / total

# Calcul des fréquences marginales des lignes (f_i.)
f_i = np.sum(f_ij, axis=1)

# Calcul des fréquences marginales des colonnes (f_.j)
f_j = np.sum(f_ij, axis=0)

# Profil par ligne (FF)
FF = f_ij / f_i[:, np.newaxis]

# Matrices diagonales
D_n = np.diag(f_i)
D_n_1 = np.diag(1 / f_i)
Fprim = FF.T
D_p = np.diag(f_j)
D_p_1 = np.diag(1 / f_j)

# Calcul de FD_n_1
FD_n_1 = np.dot(D_n_1, FF)


# Calcul de A_at
A_at = np.dot(Fprim, np.dot(D_n_1, FF))


# Calcul de L1
L1 = np.diag(f_j**(-0.5))

# Calcul de A
A = np.dot(L1, np.dot(A_at, L1))

# Calcul de S
D_p_half_inv = np.diag(1 / np.sqrt(f_j))
S = np.dot(A_at, np.dot(D_p_half_inv, D_p_half_inv))

# Calcul des valeurs propres et vecteurs propres
eigen_results_A = np.linalg.eig(A)
eigen_results_S = np.linalg.eig(S)
print(eigen_results_A,"####", eigen_results_S)
