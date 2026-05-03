#!/usr/bin/env python3
"""
Script de comparaison des données de température pour différentes géométries
de disques de frein (Pogacar, Van Aert).

Charge les fichiers NPZ générés par main_diffusion_2d.py et affiche les résultats
côte à côte avec noms professionnels.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Géométries à comparer
geometries = ["Pogi", "WVA"]

# Mapping des noms : fichiers -> affichage professionnel
name_mapping = {
    "Pogi": "Pogacar",
    "WVA": "Van Aert"
}

# Couleurs et styles pour chaque géométrie
colors = {"Pogi": "blue", "WVA": "red"}
linestyles = {"Pogi": "-", "WVA": "--"}

# Charger les données
data = {}
for geom in geometries:
    filename = f"temperature_data_{geom}.npz"
    filepath = Path(__file__).parent / filename
    
    if filepath.exists():
        loaded = np.load(filepath, allow_pickle=True)
        data[geom] = {
            'temps': loaded['temps'],
            'max_temp': loaded['max_temp'],
            'mean_temp_surface': loaded['mean_temp_surface'],
        }
        display_name = name_mapping.get(geom, geom)
        print(f"✓ Données chargées pour {display_name} : {len(loaded['temps'])} points temporels")
    else:
        print(f"✗ Fichier non trouvé : {filename}")

if len(data) == 0:
    print("\nAucune donnée trouvée. Avez-vous lancé main_diffusion_2d.py pour générer les fichiers NPZ ?")
    print("Exemples : ")
    print("  python main_diffusion_2d.py --model Pogi")
    print("  python main_diffusion_2d.py --model WVA")
    exit(1)

# Créer les graphiques de comparaison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Graphique 1 : Température maximale
ax = axes[0, 0]
for geom in data.keys():
    display_name = name_mapping.get(geom, geom)
    ax.plot(data[geom]['temps'], data[geom]['max_temp'], 
            color=colors[geom], linestyle=linestyles[geom], 
            linewidth=2.5, label=display_name, marker='o', markersize=3, alpha=0.7)
ax.set_xlabel('Temps (s)', fontsize=12)
ax.set_ylabel('Température maximale (°C)', fontsize=12)
ax.set_title('Comparaison : Température maximale', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

# Graphique 2 : Température moyenne de surface
ax = axes[0, 1]
for geom in data.keys():
    display_name = name_mapping.get(geom, geom)
    ax.plot(data[geom]['temps'], data[geom]['mean_temp_surface'], 
            color=colors[geom], linestyle=linestyles[geom], 
            linewidth=2.5, label=display_name, marker='s', markersize=3, alpha=0.7)
ax.set_xlabel('Temps (s)', fontsize=12)
ax.set_ylabel('Température moyenne surface (°C)', fontsize=12)
ax.set_title('Comparaison : Température moyenne de surface', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

# Graphique 3 : Différence de température maximale
if len(data) == 2:
    ax = axes[1, 0]
    geoms_list = list(data.keys())
    display_name_1 = name_mapping.get(geoms_list[0], geoms_list[0])
    display_name_2 = name_mapping.get(geoms_list[1], geoms_list[1])
    diff_max_temp = data[geoms_list[0]]['max_temp'] - data[geoms_list[1]]['max_temp']
    ax.plot(data[geoms_list[0]]['temps'], diff_max_temp, 
            color='green', linewidth=2.5, marker='o', markersize=3, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1)
    ax.set_xlabel('Temps (s)', fontsize=12)
    ax.set_ylabel(f'Différence T_max ({display_name_1} - {display_name_2}) (°C)', fontsize=12)
    ax.set_title('Différence de température maximale', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Graphique 4 : Différence de température moyenne surface
    ax = axes[1, 1]
    diff_mean_temp = data[geoms_list[0]]['mean_temp_surface'] - data[geoms_list[1]]['mean_temp_surface']
    ax.plot(data[geoms_list[0]]['temps'], diff_mean_temp, 
            color='purple', linewidth=2.5, marker='s', markersize=3, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1)
    ax.set_xlabel('Temps (s)', fontsize=12)
    ax.set_ylabel(f'Différence T_moy ({display_name_1} - {display_name_2}) (°C)', fontsize=12)
    ax.set_title('Différence de température moyenne surface', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
else:
    # Si une seule géométrie, afficher les deux courbes dans un même graphique
    ax = axes[1, 0]
    geom = list(data.keys())[0]
    display_name = name_mapping.get(geom, geom)
    ax.plot(data[geom]['temps'], data[geom]['max_temp'], 
            color='blue', linewidth=2.5, label='T_max', marker='o', markersize=3, alpha=0.7)
    ax.plot(data[geom]['temps'], data[geom]['mean_temp_surface'], 
            color='red', linewidth=2.5, label='T_moy surface', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Temps (s)', fontsize=12)
    ax.set_ylabel('Température (°C)', fontsize=12)
    ax.set_title(f'Données pour {display_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('comparison_geometries.png', dpi=150, bbox_inches='tight')
print("\n✓ Graphique de comparaison sauvegardé : comparison_geometries.png")
plt.show()

# Afficher un résumé des résultats
print("\n" + "="*60)
print("RÉSUMÉ DES RÉSULTATS")
print("="*60)
for geom in data.keys():
    display_name = name_mapping.get(geom, geom)
    print(f"\n{display_name}:")
    print(f"  Température maximale finale : {data[geom]['max_temp'][-1]:.2f}°C")
    print(f"  Température moyenne finale : {data[geom]['mean_temp_surface'][-1]:.2f}°C")
    print(f"  Temps simulé : {data[geom]['temps'][-1]:.2f}s")

if len(data) == 2:
    geoms_list = list(data.keys())
    display_name_1 = name_mapping.get(geoms_list[0], geoms_list[0])
    display_name_2 = name_mapping.get(geoms_list[1], geoms_list[1])
    print(f"\nDifférence ({display_name_1} - {display_name_2}):")
    print(f"  T_max finale : {data[geoms_list[0]]['max_temp'][-1] - data[geoms_list[1]]['max_temp'][-1]:+.2f}°C")
    print(f"  T_moy finale : {data[geoms_list[0]]['mean_temp_surface'][-1] - data[geoms_list[1]]['mean_temp_surface'][-1]:+.2f}°C")
