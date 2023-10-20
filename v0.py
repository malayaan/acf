import numpy as np
import matplotlib.pyplot as plt

# Données : table de contingence
data_matrix_updated = np.array([
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 10]
])

# Calculer la matrice des profils (fréquences relatives)
total_updated = np.sum(data_matrix_updated)
profiles_updated = data_matrix_updated / total_updated

# Calculer la matrice de Burt
n, m = profiles_updated.shape
burt_matrix = np.zeros((m, m))
for j in range(m):
    for j_prime in range(m):
        for i in range(n):
            burt_matrix[j, j_prime] += profiles_updated[i, j] * profiles_updated[i, j_prime] / (np.sum(profiles_updated[i, :]) * np.sum(profiles_updated[:, j_prime]))

# Centrer la matrice de Burt
column_means_burt = np.mean(burt_matrix, axis=0, keepdims=True)
overall_mean_burt = np.mean(burt_matrix)
centered_burt_matrix = burt_matrix - column_means_burt - column_means_burt.T + overall_mean_burt

# Calculer les vecteurs propres et les valeurs propres de la matrice de covariance
eigenvalues_burt, eigenvectors_burt = np.linalg.eig(centered_burt_matrix)
sorted_indices_burt = np.argsort(eigenvalues_burt)[::-1]
eigenvalues_burt = eigenvalues_burt[sorted_indices_burt]
eigenvectors_burt = eigenvectors_burt[:, sorted_indices_burt]

# Projeter les données dans l'espace des deux premiers vecteurs propres
factor_scores_burt = np.dot(centered_burt_matrix, eigenvectors_burt[:, :2])

# Calculer les marges à nouveau
row_sums = np.sum(data_matrix_updated, axis=1)
column_sums = np.sum(data_matrix_updated, axis=0)

# Calculer la matrice des données attendues
expected_data_matrix = np.outer(row_sums, column_sums) / np.sum(data_matrix_updated)

# Affichage
plt.figure(figsize=(10, 8))
unique_perceived_flavors = ['Perçu acide', 'Perçu amer', 'Perçu sucré']
for j, perceived_flavor in enumerate(unique_perceived_flavors):
    plt.scatter(factor_scores_burt[j, 0], factor_scores_burt[j, 1], label=perceived_flavor, s=100)
    plt.annotate(perceived_flavor, (factor_scores_burt[j, 0], factor_scores_burt[j, 1]), fontsize=12)
plt.axhline(0, color='grey', linewidth=0.5)
plt.axvline(0, color='grey', linewidth=0.5)
plt.title('Analyse des Correspondances Multiples (ACM) en utilisant la matrice de Burt')
plt.xlabel('Première dimension')
plt.ylabel('Deuxième dimension')
plt.legend()
plt.grid(True)
plt.show()
