import numpy as np
import matplotlib.pyplot as plt

def read_data_from_file(file_name):
    """Lire les données du fichier."""
    with open(file_name, 'r') as file:
        lines = file.readlines()[1:]
        return [line.strip().split(', ') for line in lines]

def create_contingency_table(data):
    """Créer la table de contingence."""
    unique_values_1 = sorted(list(set([pair[0] for pair in data])))
    unique_values_2 = sorted(list(set([pair[1] for pair in data])))

    table = np.zeros((len(unique_values_1), len(unique_values_2)))
    for pair in data:
        i = unique_values_1.index(pair[0])
        j = unique_values_2.index(pair[1])
        table[i, j] += 1

    return table, unique_values_1, unique_values_2

def compute_profiles(contingency_table):
    """Calculer les profils."""
    return contingency_table / contingency_table.sum()

def compute_burt_matrix(profiles):
    """Calculer la matrice de Burt avec la formule du cours en utilisant une optimisation vectorielle."""
    # Calculating row and column sums
    row_sums = profiles.sum(axis=1)
    col_sums = profiles.sum(axis=0)
    
    # Number of rows and columns
    n_rows, n_cols = profiles.shape
    
    # Initializing the Burt matrix with zeros
    burt_matrix = np.zeros((n_cols, n_cols))
    
    # Filling the Burt matrix using the provided formula
    for j in range(n_cols):
        for j_prime in range(n_cols):
            burt_matrix[j, j_prime] = np.sum(
                (profiles[:, j] * profiles[:, j_prime]) / 
                (row_sums * np.sqrt(col_sums[j] * col_sums[j_prime]))
            )
    
    return burt_matrix

def compute_eigen_decomposition(matrix):
    """Calculer la décomposition propre."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenorder = np.argsort(eigenvalues)[::-1]
    return eigenvalues[eigenorder], eigenvectors[:, eigenorder]

def project_on_eigenvectors(matrix, eigenvectors, n_dimensions=2):
    """Projeter la matrice sur les vecteurs propres fournis."""
    return np.dot(matrix, eigenvectors[:, :n_dimensions])

def plot_data(scores_rows, scores_columns, labels_rows, labels_columns):
    """Tracer les données dans l'espace réduit."""
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

def quality_of_projection(eigenvalues):
    """Calculer la qualité de la projection."""
    total_inertia = sum(eigenvalues)
    explained_inertia = [eig / total_inertia for eig in eigenvalues]
    return np.cumsum(explained_inertia)

if __name__ == "__main__":
    # Étape 1: Transformer les données
    data = read_data_from_file('sondage_irrégulier.txt')
    contingency_table, unique_values_1, unique_values_2 = create_contingency_table(data)
    contingency_table, unique_values_1, unique_values_2 = np.array([[10, 0, 0],[0, 9, 1],[0, 3, 7]]),['acide', 'amer', 'sucré'] ,['percu acide', 'percu amer', 'percu sucré']
    profiles = compute_profiles(contingency_table)
    print(profiles)
    
    # Étape 2: Réduction et projection
    burt_matrix = compute_burt_matrix(np.array(profiles))
    centered_burt_rows = burt_matrix - burt_matrix.mean(axis=1, keepdims=True)
    centered_burt_columns = burt_matrix - burt_matrix.mean(axis=0, keepdims=True)

    # Calculer les projections des données centrées sur les lignes
    eigenvalues_rows, eigenvectors_rows = compute_eigen_decomposition(centered_burt_rows)
    scores_rows = project_on_eigenvectors(centered_burt_rows, eigenvectors_rows)

    # Calculer les projections des données centrées sur les colonnes
    eigenvalues_columns, eigenvectors_columns = compute_eigen_decomposition(centered_burt_columns)
    scores_columns = project_on_eigenvectors(centered_burt_columns, eigenvectors_columns)

    # Étape 3: Tracer les données
    plot_data(scores_rows, scores_columns, unique_values_1, unique_values_2)
    
    # Étape 4: Qualité de la projection
    cascade = quality_of_projection(eigenvalues_rows)
    print("Cascade de valeurs propres (rows):", cascade)

    cascade_columns = quality_of_projection(eigenvalues_columns)
    print("Cascade de valeurs propres (columns):", cascade_columns)