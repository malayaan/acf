# Exécution du code initial pour préparer les données et les matrices nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Table de contingence donnée
table_contingence = pd.DataFrame({
    'Perçu sucré': [10, 0, 0],
    'Perçu acide': [0, 10, 0],
    'Perçu amer': [0, 0, 10]
}, index=['sucré', 'acide', 'amer'])

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

# Calcul de FD_n_1
FD_n_1 = np.dot(D_n_1, F)

# Calcul de A_at : la matrice d'inertie
A_at = np.dot(Fprim, np.dot(D_n_1, F))

# Calcul de L1
L1 = np.diag(f_j**(-1/2))

# Calcul de A : matrice d'inertie transformée
A = np.dot(L1, np.dot(A_at, L1))

# Calcul des valeurs propres et décomposition en valeurs singulières de A
valeurs_propres, vecteurs_propres = np.linalg.eig(A)

# Trier les valeurs propres et obtenir les indices pour les deux plus grandes
indices = np.argsort(valeurs_propres)[::-1][:2]

# Sélection des deux plus grandes valeurs propres et des vecteurs propres associés
plus_grandes_valeurs = valeurs_propres[indices]
plus_grands_vecteurs = np.array([vecteurs_propres[indices[0]],vecteurs_propres[indices[1]]])

# Calcul des vecteurs propres de S et T
u = np.dot(D_n_1, plus_grands_vecteurs.T).T
v = np.zeros((2, A.shape[0]))
for alpha in range(2):
    u_alpha = u[alpha]
    lambda_alpha = plus_grandes_valeurs[alpha]
    v[alpha] = (1 / np.sqrt(lambda_alpha)) * np.dot(F, np.dot(D_p_1, u_alpha))

# Calcul des coordonnées factorielles pour le profil des lignes et le profil des colonnes
psi = np.zeros((2, A.shape[0]))
phi = np.zeros((2, A.shape[0]))

for alpha in range(2):
    psi[alpha] = np.dot(D_n_1, np.dot(F, np.dot(D_p_1, u[alpha])))
    phi[alpha] = np.dot(D_p_1, np.dot(F, np.dot(D_n_1, v[alpha])))

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
plt.show()
