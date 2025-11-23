#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération de graphes des spins nucléaires à partir de ground_states.csv (IAEA).

Pour chaque élément (Z = 1..118), le script produit:
  - un graphique 2D J(A) (spin signé en fonction du numéro de masse A),
  - un graphique 3D léger:
      x = A (isotopes)
      y = indice de niveau (0 = état fondamental)
      z = spin signé J (parité + → +J, parité − → −J)

Notes:
  - A est calculé comme A = Z + N à partir des colonnes 'z' et 'n'.
  - La parité est encodée dans la colonne 'jp' (ex: '3/2-', '2+', '(1- 2-)').
    On l'interprète en Jπ → ±J:
        '3/2+'  → +1.5
        '3/2-'  → -1.5
        '0+'    → 0.0
        '0-'    → 0.0 (parité différente mais spin nul)
  - Les cas ambigus (ex: '(1- 2-)', '5/2- 7/2-') sont résolus en prenant
    la première valeur de spin indiquée.

Usage:
  python plot_spins_iaea.py ground_states.csv -o plots_spins

Cela créera:
  plots_spins/2D/Z001_H_spins_2D.png, ...
  plots_spins/3D/Z001_H_spins_3D.png, ...
"""

import argparse
import math
import os
import re
from typing import Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – nécessaire pour projection='3d'
import pandas as pd
import numpy as np

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Parsing du spin JP → valeur signée
# ---------------------------------------------------------------------------

def parse_spin_jp(jp: Optional[str]) -> float:
    """
    Convertit une chaîne 'jp' (Jπ) de l'IAEA en spin signé.

    Conventions:
      - On retourne NaN si la valeur est manquante ou illisible.
      - On traduit Jπ en ±J:
          '3/2+'     → +1.5
          '3/2-'     → -1.5
          '(1/2+)'   → +0.5
          '(1- 2-)'  → -1.0   (on prend le premier terme '1-')
          '5/2(+)'   → +2.5
          '5/2(-)'   → -2.5
      - S'il n'y a pas de signe de parité explicite, on suppose parité positive.

    Cela permet d'obtenir des valeurs < -1/2, etc., comme souhaité.
    """
    if not isinstance(jp, str):
        return math.nan

    s = jp.strip()
    if not s:
        return math.nan

    # Normaliser les espaces
    s = re.sub(r'\s+', ' ', s)

    # On prend seulement le premier "candidat" si plusieurs sont proposés
    # Exemple: "(1- 2-)" → token = "1-"
    token = s.split(' ')[0]

    # Normaliser les formes '(+)' et '(-)' en '+' / '-'
    token = token.replace('(+)', '+').replace('(-)', '-')

    # Retirer des parenthèses isolées autour du token si présentes
    if token.startswith('(') and token.endswith(')') and len(token) > 2:
        token = token[1:-1]

    # Détecter la parité en suffixe
    parity_sign = 1.0
    if token.endswith('+'):
        parity_sign = 1.0
        core = token[:-1]
    elif token.endswith('-'):
        parity_sign = -1.0
        core = token[:-1]
    else:
        core = token

    core = core.strip().strip('()')
    if not core:
        return math.nan

    # Conversion "3/2" → 1.5 ou "3" → 3.0
    try:
        if '/' in core:
            num_str, den_str = core.split('/', 1)
            spin_mag = float(num_str) / float(den_str)
        else:
            spin_mag = float(core)
    except Exception:
        return math.nan

    return parity_sign * spin_mag


# ---------------------------------------------------------------------------
# Génération des graphes pour un élément
# ---------------------------------------------------------------------------

def make_plots_for_element(group: pd.DataFrame, out_dir: str) -> bool:
    """
    Crée les deux graphes (2D et 3D) pour un élément donné (groupé par Z),
    avec interpolation curviligne si possible (CubicSpline).
    """
    z_val = int(group["z"].iloc[0])

    # Symbol chimique
    sym_series = group["symbol"].dropna()
    symbol = sym_series.iloc[0].strip() if len(sym_series) > 0 else f"Z{z_val}"

    # On ne garde que les isotopes dont le spin est défini
    sub = group[group["spin_signed"].notna()].copy()
    if sub.empty:
        print(f"[Z={z_val:3d}] {symbol:>3} : aucun spin jp défini, graphes ignorés.")
        return False

    # Tri par A
    sub = sub.sort_values("A")
    xs = sub["A"].values.astype(float)
    zs = sub["spin_signed"].values.astype(float)
    ys = np.zeros_like(xs)  # niveau 0 = état fondamental

    # Préparation interpolation curviligne
    use_spline = HAS_SCIPY and len(xs) >= 3 and np.all(np.diff(xs) > 0)
    if use_spline:
        # grille dense pour un rendu lisse
        x_dense = np.linspace(xs.min(), xs.max(), 200)
        spline = CubicSpline(xs, zs)
        z_dense = spline(x_dense)
        y_dense = np.zeros_like(x_dense)
    else:
        x_dense = xs
        z_dense = zs
        y_dense = ys

    # Répertoires de sortie
    out_2d = os.path.join(out_dir, "2D")
    out_3d = os.path.join(out_dir, "3D")
    os.makedirs(out_2d, exist_ok=True)
    os.makedirs(out_3d, exist_ok=True)

    prefix = f"Z{z_val:03d}_{symbol}"

    # -------- Graphique 2D : J(A) --------
    fig2d, ax2d = plt.subplots(figsize=(7, 4))

    # courbe lissée (si possible)
    ax2d.plot(x_dense, z_dense, linestyle="-")
    # points expérimentaux
    ax2d.plot(xs, zs, marker="o", linestyle="")

    ax2d.set_xlabel("Numéro de masse A")
    ax2d.set_ylabel("Spin signé J (Jπ → ±J)")
    ax2d.set_title(f"{symbol} (Z={z_val}) – spins des états fondamentaux")
    ax2d.axhline(0.0, linewidth=0.8)
    fig2d.tight_layout()
    out_path_2d = os.path.join(out_2d, f"{prefix}_spins_2D.png")
    fig2d.savefig(out_path_2d, dpi=150)
    plt.close(fig2d)

    # -------- Graphique 3D : courbe en légère perspective --------
    fig3d = plt.figure(figsize=(7, 4))
    ax3d = fig3d.add_subplot(111, projection="3d")

    # courbe lissée
    ax3d.plot(x_dense, y_dense, z_dense, linestyle="-")
    # points expérimentaux
    ax3d.scatter(xs, ys, zs, marker="o")

    ax3d.set_xlabel("Numéro de masse A")
    ax3d.set_ylabel("Indice de niveau (0 = gs)")
    ax3d.set_zlabel("Spin signé J (Jπ → ±J)")
    ax3d.set_title(f"{symbol} (Z={z_val}) – spins en 3D légère")

    ax3d.view_init(elev=25, azim=-60)

    fig3d.tight_layout()
    out_path_3d = os.path.join(out_3d, f"{prefix}_spins_3D.png")
    fig3d.savefig(out_path_3d, dpi=150)
    plt.close(fig3d)

    print(f"[Z={z_val:3d}] {symbol:>3} : graphes 2D/3D (lissés) générés.")
    return True



# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Génère, à partir de ground_states.csv (IAEA), "
                    "des graphes 2D et 3D des spins nucléaires par élément."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="ground_states.csv",
        help="Chemin vers ground_states.csv (défaut: ./ground_states.csv)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots_spins",
        help="Répertoire de sortie pour les images (défaut: plots_spins)."
    )
    args = parser.parse_args()

    csv_path = args.csv
    out_dir = args.output_dir

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable: {csv_path}")

    # Lecture du CSV IAEA
    df = pd.read_csv(csv_path)

    # Ajout du numéro de masse A = Z + N
    if not {"z", "n", "jp"}.issubset(df.columns):
        raise ValueError(
            "Le CSV ne contient pas les colonnes attendues 'z', 'n', 'jp'. "
            "Vérifie que tu utilises bien le fichier ground_states.csv "
            "fourni par l'API REST de l'IAEA."
        )

    df["A"] = df["z"] + df["n"]

    # Parsing du spin signé à partir de 'jp'
    df["spin_signed"] = df["jp"].apply(parse_spin_jp)

    # On ignore Z=0 (neutron libre) pour les graphes par élément
    df = df[df["z"] > 0].copy()

    # Groupement par élément (numéro atomique Z)
    n_elements = 0
    n_plots = 0
    for z_val, group in df.groupby("z"):
        created = make_plots_for_element(group, out_dir=out_dir)
        if created:
            n_elements += 1
            n_plots += 2

    print()
    print(f"Terminé : {n_elements} éléments tracés, {n_plots} images générées.")
    print("Les graphes se trouvent dans :")
    print(f"  {os.path.join(out_dir, '2D')}")
    print(f"  {os.path.join(out_dir, '3D')}")
    print()
    print("Remarque : si certains éléments n'ont pas de spin jp défini dans "
          "ground_states.csv, leurs graphes ne sont pas générés.")


if __name__ == "__main__":
    main()
