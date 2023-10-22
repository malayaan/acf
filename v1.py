import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_excel_data(file_name, column_1, column_2):
    """Load data from an Excel file and extract specified columns."""
    excel_data = pd.read_excel(file_name)
    return excel_data[[column_1, column_2]].dropna().values.tolist()

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

def compute_profiles_and_inertia(table):
    """Compute the matrix of profiles and inertia from a contingency table."""
    # Calculate the matrix of profiles FF
    total = table.values.sum()
    F = table / total

    # Calculate D_n and D_n_1
    f_i = F.sum(axis=1)
    D_n = np.diag(f_i)
    D_n_1 = np.diag(1/f_i)

    # Calculate Fprim, D_p, and D_p_1
    Fprim = F.T
    f_j = F.sum(axis=0)
    D_p = np.diag(f_j)
    D_p_1 = np.diag(1/f_j)

    # Calculate A_at and A
    A_at = np.dot(Fprim, np.dot(D_n_1, F))
    L1 = np.diag(f_j**(-1/2))
    A = np.dot(L1, np.dot(A_at, L1))

    return F, D_n_1, D_p_1, A


def compute_singular_decomposition(A):
    """Compute the singular value decomposition of matrix A."""
    valeurs_propres, vecteurs_propres = np.linalg.eig(A)
    indices = np.argsort(valeurs_propres)[::-1][:2]
    plus_grandes_valeurs = valeurs_propres[indices]
    plus_grands_vecteurs = np.array([vecteurs_propres[indices[0]], vecteurs_propres[indices[1]]])
    return plus_grandes_valeurs, plus_grands_vecteurs


def compute_coordinates(F, D_n_1, D_p_1, plus_grandes_valeurs, plus_grands_vecteurs):
    """Compute the factorial coordinates for the row and column profiles."""
    f_j = F.sum(axis=0)
    u = np.dot(np.diag(f_j**(1/2)), plus_grands_vecteurs.T).T
    v = np.zeros((2, F.shape[0]))

    for alpha in range(2):
        lambda_alpha = plus_grandes_valeurs[alpha]
        v[alpha] = (1 / np.sqrt(lambda_alpha)) * np.dot(F, np.dot(D_p_1, u[alpha]))

    psi = np.zeros((2, F.shape[0]))
    phi = np.zeros((2, F.shape[1]))

    for alpha in range(2):
        lambda_alpha = plus_grandes_valeurs[alpha]
        psi[alpha] = np.sqrt(lambda_alpha) * np.dot(D_n_1, v[alpha])
        phi[alpha] = np.sqrt(lambda_alpha) * np.dot(D_p_1, u[alpha])

    return psi, phi


def plot_profiles(psi, phi, table):
    """Plot the row and column profiles on a factorial plane."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotting rows
    ax.scatter(psi[0], psi[1], marker='o', color='blue', label='Lignes')
    for i, label in enumerate(table.index):
        ax.text(psi[0, i], psi[1, i], label, color='blue')

    # Plotting columns
    ax.scatter(phi[0], phi[1], marker='s', color='red', label='Colonnes')
    for j, label in enumerate(table.columns):
        ax.text(phi[0, j], phi[1, j], label, color='red')

    # Graph settings
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_title("Représentation des profils des lignes et des colonnes")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def main_analysis(file_name, column_1, column_2):
    # Load data from Excel file
    data_excel = load_excel_data(file_name, column_1, column_2)

    # Create the contingency table
    contingency_table_excel, row_labels_excel, column_labels_excel = create_contingency_table(data_excel)
    table_contingence = pd.DataFrame(contingency_table_excel, index=row_labels_excel, columns=column_labels_excel)

    # Compute profiles and inertia matrices
    F, D_n_1, D_p_1, A = compute_profiles_and_inertia(table_contingence)

    # Compute the singular value decomposition
    plus_grandes_valeurs, plus_grands_vecteurs = compute_singular_decomposition(A)

    # Compute the factorial coordinates
    psi, phi = compute_coordinates(F, D_n_1, D_p_1, plus_grandes_valeurs, plus_grands_vecteurs)

    # Plot the results
    plot_profiles(psi, phi, table_contingence)
    

# Call the main function with the correct path to the Excel file
main_analysis('TP_AFC_majeur1718_travail (1).xlsx', " temps travail", "Qualite vie")

