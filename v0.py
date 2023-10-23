# Exécution du code initial pour préparer les données et les matrices nécessaires
import numpy as np
import pandas as pd
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

"""
# Lire le fichier Excel
excel_data = pd.read_excel('TP_AFC_majeur1718_travail (1).xlsx')

# Extraire les colonnes " temps travail" et "Qualite vie" avec les noms corrects
extracted_data_excel = excel_data[[" temps travail", "Qualite vie"]].dropna().values.tolist()

# Créer la table de contingence pour " temps travail" et "Qualite vie"
contingency_table_excel, row_labels_excel, column_labels_excel = create_contingency_table(extracted_data_excel)

# Convertir la table de contingence en DataFrame pour une meilleure visualisation
table_contingence = pd.DataFrame(contingency_table_excel, index=row_labels_excel, columns=column_labels_excel)
"""
"""
# Lire les données du fichier
data = read_data_from_file('sondage.txt')

# Créer la table de contingence
contingency_table, row_labels, column_labels = create_contingency_table(data)

# Convertir la table de contingence en DataFrame pour une meilleure visualisation
table_contingence= pd.DataFrame(contingency_table, index=row_labels, columns=column_labels)

print(table_contingence)

"""
# Table de contingence donnée
table_contingence = pd.DataFrame({
    'percu acide': [9, 3, 0],
    'percu amer': [1, 7, 0],
    'percu sucré': [0,0,10]
}, index=['acide','amer','sucré'])


# Calcul de la matrice des profils FF
total = table_contingence.values.sum()
F = table_contingence / total

# Calcul de D_n : matrice diagonale des fréquences marginales des lignes
f_i = F.sum(axis=1)
D_n = np.diag(f_i)

# Calcul de D_n_1 : matrice diagonale inverse des fréquences marginales des lignes
D_n_1 = np.diag(1/f_i)

# Calcul de Fprim : transposition de la matrice des profils
Fprim = F.T

# Calcul de D_p : matrice diagonale des fréquences marginales des colonnes
f_j = F.sum(axis=0)
D_p = np.diag(f_j)

# Calcul de D_p_1 : matrice diagonale inverse des fréquences marginales des colonnes
D_p_1 = np.diag(1/f_j)

# Calcul de A_at : la matrice d'inertie
A_at = np.dot(Fprim, np.dot(D_n_1, F))

# Calcul de A : matrice d'inertie transformée
A = np.dot(np.diag(f_j**(-1/2)), np.dot(A_at, np.diag(f_j**(-1/2))))

# Calcul des valeurs propres et décomposition en valeurs singulières de A
valeurs_propres, vecteurs_propres = np.linalg.eig(A)

# Trier les valeurs propres et obtenir les indices pour les deux plus grandes
#indices = np.argsort(valeurs_propres)[::-1][:2]

# Éliminer les doublons et trier les valeurs propres
valeurs_propres_uniques = np.unique(valeurs_propres)
indices_uniques = np.argsort(valeurs_propres_uniques)[::-1]

# Obtenir les indices pour les deux plus grandes valeurs propres distinctes
indices = indices_uniques[:2]

# Sélection des deux plus grandes valeurs propres et des vecteurs propres associés
plus_grandes_valeurs = valeurs_propres[indices]
plus_grands_vecteurs = np.array([vecteurs_propres[indices[0]],vecteurs_propres[indices[1]]])
print(A)
print(vecteurs_propres)
print(valeurs_propres)

# Calcul des vecteurs propres de S et T
u = np.dot(np.diag(f_j**(1/2)), plus_grands_vecteurs.T).T
v = np.zeros((2, F.shape[0]))
for alpha in range(2):
    lambda_alpha = plus_grandes_valeurs[alpha]
    v[alpha] = (1 / np.sqrt(lambda_alpha)) * np.dot(F, np.dot(D_p_1, u[alpha]))

# Calcul des coordonnées factorielles pour le profil des lignes et le profil des colonnes
psi = np.zeros((2, F.shape[0]))
phi = np.zeros((2, F.shape[1]))
for alpha in range(2):
    lambda_alpha = plus_grandes_valeurs[alpha]
    psi[alpha] = np.sqrt(lambda_alpha) *np.dot(D_n_1, v[alpha])
    phi[alpha] = np.sqrt(lambda_alpha) *np.dot(D_p_1, u[alpha])

# Tracer les résultats avec les coordonnées factorielles
fig, ax = plt.subplots(figsize=(10, 8))

# Pour les lignes
ax.scatter(psi[0], psi[1], marker='o', color='blue', label='Lignes')

# Pour les colonnes
ax.scatter(phi[0], phi[1], marker='s', color='red', label='Colonnes')

# Ajout des étiquettes pour chaque point
for i, label in enumerate(table_contingence.index):
    ax.text(psi[0, i], psi[1, i], label, color='blue')

for j, label in enumerate(table_contingence.columns):
    ax.text(phi[0, j], phi[1, j], label, color='red')

# Configurer le graphique
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)
ax.set_title("Représentation des profils des lignes et des colonnes")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()

# Montrer le graphique
plt.tight_layout()
#plt.show()