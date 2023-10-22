import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Table de contingence donnée
table_contingence = pd.DataFrame({
    'Perçu acide': [10, 0, 0],
    'Perçu amer': [0, 9, 1],
    'Perçu sucré': [0, 3, 7]
}, index=['acide', 'amer', 'sucré'])

# Calcul de la matrice des profils FF
total = table_contingence.values.sum()
profil_ij = table_contingence / total

# Calcul de D_n : matrice diagonale des fréquences marginales des lignes
f_i = table_contingence.sum(axis=1) / total
D_n = np.diag(f_i)

# Calcul de D_n_1 : matrice diagonale inverse des fréquences marginales des lignes
D_n_1 = np.diag(1/f_i)

# Calcul de Fprim : transposition de la matrice des profils
Fprim = profil_ij.T

# Calcul de D_p : matrice diagonale des fréquences marginales des colonnes
f_j = table_contingence.sum(axis=0) / total
D_p = np.diag(f_j)

# Calcul de D_p_1 : matrice diagonale inverse des fréquences marginales des colonnes
D_p_1 = np.diag(1/f_j)

# Calcul de FD_n_1
FD_n_1 = np.dot(D_n_1, profil_ij)

# Calcul de A_at : la matrice d'inertie
A_at = np.dot(Fprim, np.dot(D_n_1, profil_ij))

# Calcul de L1
L1 = np.diag(f_j**(-1/2))

# Calcul de A : matrice d'inertie transformée
A = np.dot(L1, np.dot(A_at, L1))

# Calcul des valeurs propres et décomposition en valeurs singulières de A
valeurs_propres = np.linalg.eigvals(A)
u, s, vh = np.linalg.svd(A)

# Trier les valeurs propres et obtenir les indices pour les deux plus grandes
indices = np.argsort(valeurs_propres)[::-1][:2]

# Sélection des deux plus grandes valeurs propres et des vecteurs propres associés de la matrice vh
plus_grandes_valeurs = valeurs_propres[indices]
plus_grands_vecteurs = vh[indices, :]

# Calcul des coordonnées des lignes et des colonnes dans l'espace factoriel
coordonnees_lignes = np.dot(FD_n_1, plus_grands_vecteurs.T)
coordonnees_colonnes = np.dot(profil_ij.T, np.dot(D_n_1, plus_grands_vecteurs.T))

# Tracer les résultats
fig, ax = plt.subplots(figsize=(10, 8))

# Tracer les points de ligne (Vraie-Saveur)
for i, etiquette in enumerate(table_contingence.index):
    ax.scatter(coordonnees_lignes[i, 0], coordonnees_lignes[i, 1], marker='o', color='r', s=100)
    ax.text(coordonnees_lignes[i, 0] + 0.05, coordonnees_lignes[i, 1] + 0.05, etiquette, fontsize=12)

# Tracer les points de colonne (Saveur-Perçue)
for i, etiquette in enumerate(table_contingence.columns):
    ax.scatter(coordonnees_colonnes[i, 0], coordonnees_colonnes[i, 1], marker='s', color='b', s=100)
    ax.text(coordonnees_colonnes[i, 0] + 0.05, coordonnees_colonnes[i, 1] + 0.05, etiquette, fontsize=12)

ax.axhline(0, color='grey', linestyle='--')
ax.axvline(0, color='grey', linestyle='--')
ax.set_title("Plan des Valeurs Propres (AFC)")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")

plt.show()
