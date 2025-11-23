#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spins_levels_from_catalog.py

Génère, à partir d'un CSV unifié (ground + niveaux + extras)
produit par build_nuclear_levels_catalog.py, deux graphes par élément
(Z = 1..118 environ) :

  1) Graphique 2D (par élément) :
       - x = numéro de masse A
       - y = spin signé J (Jπ → ±J)
     avec :
       * courbe lissée (spline cubique si SciPy dispo) passant par les
         spins des états fondamentaux (is_ground = True)
       * points pour tous les niveaux (ground + excités)

  2) Graphique 3D (par élément) :
       - x = numéro de masse A
       - y = énergie du niveau (keV)
       - z = spin signé J (Jπ → ±J)
     vue en légère perspective.

Le CSV d'entrée doit contenir (au minimum) les colonnes :
  - z, a, symbol
  - energy_keV ou energy_kev
  - jpi (ou jp si jpi absent)
  - is_ground (booléen ou équivalent)

Dépendances :
    pip install numpy pandas matplotlib
    pip install scipy        # pour l'interpolation spline (recommandé)

Usage :
    python plot_spins_levels_from_catalog.py \
        nuclear_levels_catalog.csv \
        -o plots_levels

Cela créera :
    plots_levels/2D/Z001_H_spins_levels_2D.png
    plots_levels/3D/Z001_H_spins_levels_3D.png
    etc.
"""

import argparse
import math
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – nécessaire pour projection='3d'

# ---------------------------------------------------------------------------
# Interpolation : SciPy optionnelle
# ---------------------------------------------------------------------------

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Parsing du spin Jπ → valeur signée
# ---------------------------------------------------------------------------

def parse_spin_jp(jp: Optional[str]) -> float:
    """
    Convertit une chaîne 'Jπ' (ou 'jp') en spin signé.

    Conventions :
      - Retourne NaN si la valeur est manquante ou illisible.
      - Traduit Jπ en ±J :
          '3/2+'     → +1.5
          '3/2-'     → -1.5
          '0+'       → 0.0
          '0-'       → 0.0 (parité différente mais spin nul)
          '(1- 2-)'  → -1.0   (prend le premier terme '1-')
          '5/2(+)'   → +2.5
          '5/2(-)'   → -2.5
      - S'il n'y a pas de signe explicite de parité, on suppose parité positive.

    Cela permet d'obtenir des valeurs éventuelles >3/2 ou < -1/2, etc.,
    telles que présentes dans les catalogues.
    """
    if not isinstance(jp, str):
        return math.nan

    s = jp.strip()
    if not s:
        return math.nan

    # normaliser les espaces
    s = re.sub(r"\s+", " ", s)

    # prendre seulement le premier "candidat" si plusieurs proposés
    token = s.split(" ")[0]

    # normaliser '(+)' et '(-)'
    token = token.replace('(+)', '+').replace('(-)', '-')

    # retirer d'éventuelles parenthèses entourant tout le token
    if token.startswith("(") and token.endswith(")") and len(token) > 2:
        token = token[1:-1]

    # détection parité
    parity_sign = 1.0
    if token.endswith("+"):
        parity_sign = 1.0
        core = token[:-1]
    elif token.endswith("-"):
        parity_sign = -1.0
        core = token[:-1]
    else:
        core = token

    core = core.strip().strip("()")
    if not core:
        return math.nan

    try:
        if "/" in core:
            num_str, den_str = core.split("/", 1)
            spin_mag = float(num_str) / float(den_str)
        else:
            spin_mag = float(core)
    except Exception:
        return math.nan

    return parity_sign * spin_mag


# ---------------------------------------------------------------------------
# Fonctions de tracé par élément
# ---------------------------------------------------------------------------

def make_plots_for_element(group: pd.DataFrame, out_dir: str, energy_col: str) -> bool:
    """
    Crée les deux graphes (2D et 3D) pour un élément donné (groupé par Z).

    Paramètres
    ----------
    group : DataFrame
        Sous-table pour un Z donné (tous isotopes + niveaux).
        Doit contenir : 'z', 'a', 'symbol', 'spin_signed', 'is_ground', energy_col.
    out_dir : str
        Répertoire racine de sortie.
    energy_col : str
        Nom de la colonne d'énergie (keV).

    Retourne
    --------
    bool
        True si au moins un graphe a été produit pour cet élément.
    """
    if group.empty:
        return False

    z_val = int(group["z"].iloc[0])

    # symbole chimique (on prend le premier non NaN)
    sym_series = group["symbol"].dropna()
    symbol = sym_series.iloc[0].strip() if len(sym_series) > 0 else f"Z{z_val}"

    # On garde seulement les lignes avec spin défini
    sub = group[group["spin_signed"].notna()].copy()
    if sub.empty:
        print(f"[Z={z_val:3d}] {symbol:>3} : aucun spin Jπ défini, graphes ignorés.")
        return False

    # Répertoires de sortie
    out_2d = os.path.join(out_dir, "2D")
    out_3d = os.path.join(out_dir, "3D")
    os.makedirs(out_2d, exist_ok=True)
    os.makedirs(out_3d, exist_ok=True)

    prefix = f"Z{z_val:03d}_{symbol}"

    # ----- 2D : spins vs A (avec interpolation sur les ground states) -----

    # Ground states (is_ground = True ou équivalent)
    # il peut y avoir des exports où is_ground n'est pas strictement bool → caster
    if "is_ground" in sub.columns:
        gs = sub[sub["is_ground"].astype(bool)].copy()
    else:
        # fallback : on suppose qu'on n'a que des ground (cas extrême)
        gs = sub.copy()

    # tri par A
    gs = gs.sort_values("a")
    xs_g = gs["a"].values.astype(float)
    zs_g = gs["spin_signed"].values.astype(float)

    # Interpolation spline sur les ground uniquement
    use_spline = (
        HAS_SCIPY
        and len(xs_g) >= 3
        and np.all(np.diff(xs_g) > 0)
    )
    if use_spline:
        x_dense = np.linspace(xs_g.min(), xs_g.max(), 200)
        spline = CubicSpline(xs_g, zs_g)
        z_dense = spline(x_dense)
    else:
        x_dense = xs_g
        z_dense = zs_g

    # Tous les niveaux (pour les points)
    sub_2d = sub.sort_values(["a", energy_col])

    fig2d, ax2d = plt.subplots(figsize=(7, 4))

    # Courbe lissée (ou non) ground
    ax2d.plot(x_dense, z_dense, linestyle="-", label="ground (interp.)" if use_spline else "ground")

    # Points ground
    ax2d.plot(xs_g, zs_g, marker="o", linestyle="", label="ground (points)")

    # Points niveaux excités (si présents)
    if "is_ground" in sub_2d.columns:
        excited = sub_2d[~sub_2d["is_ground"].astype(bool)]
    else:
        excited = pd.DataFrame()

    if not excited.empty:
        ax2d.plot(
            excited["a"].values.astype(float),
            excited["spin_signed"].values.astype(float),
            linestyle="",
            marker="x",
            label="niveaux excités"
        )

    ax2d.set_xlabel("Numéro de masse A")
    ax2d.set_ylabel("Spin signé J (Jπ → ±J)")
    ax2d.set_title(f"{symbol} (Z={z_val}) – spins par isotopes (ground + niveaux)")
    ax2d.axhline(0.0, linewidth=0.8)
    ax2d.legend(loc="best")
    fig2d.tight_layout()

    out_path_2d = os.path.join(out_2d, f"{prefix}_spins_levels_2D.png")
    fig2d.savefig(out_path_2d, dpi=150)
    plt.close(fig2d)

    # ----- 3D : A vs énergie vs spin -----

    # On garde seulement les lignes avec énergie définie
    sub_3d = sub[sub[energy_col].notna()].copy()
    if sub_3d.empty:
        print(f"[Z={z_val:3d}] {symbol:>3} : pas d'énergie définie pour les niveaux, graphe 3D ignoré.")
        return True  # 2D a été généré

    xs = sub_3d["a"].values.astype(float)
    ys = sub_3d[energy_col].values.astype(float)
    zs = sub_3d["spin_signed"].values.astype(float)

    fig3d = plt.figure(figsize=(7, 4))
    ax3d = fig3d.add_subplot(111, projection="3d")

    ax3d.scatter(xs, ys, zs, marker="o")

    ax3d.set_xlabel("Numéro de masse A")
    ax3d.set_ylabel(f"Énergie niveau ({energy_col})")
    ax3d.set_zlabel("Spin signé J (Jπ → ±J)")
    ax3d.set_title(f"{symbol} (Z={z_val}) – niveaux (A, énergie, spin)")

    # légère perspective
    ax3d.view_init(elev=25, azim=-60)

    fig3d.tight_layout()
    out_path_3d = os.path.join(out_3d, f"{prefix}_spins_levels_3D.png")
    fig3d.savefig(out_path_3d, dpi=150)
    plt.close(fig3d)

    print(f"[Z={z_val:3d}] {symbol:>3} : graphes 2D/3D générés.")
    return True


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Génère, à partir d'un catalogue CSV (ground + niveaux + extras), "
            "des graphes 2D et 3D des spins/énergies par élément."
        )
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="nuclear_levels_catalog.csv",
        help="Chemin vers le CSV unifié (défaut: nuclear_levels_catalog.csv)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots_levels",
        help="Répertoire de sortie pour les images (défaut: plots_levels)."
    )
    parser.add_argument(
        "--min-z",
        type=int,
        default=1,
        help="Z minimum à tracer (défaut: 1)."
    )
    parser.add_argument(
        "--max-z",
        type=int,
        default=118,
        help="Z maximum à tracer (défaut: 118)."
    )
    parser.add_argument(
        "--only-iaea",
        action="store_true",
        help=(
            "Si présent, ne prend que les lignes source='IAEA_LiveChart_ground' "
            "ou 'IAEA_LiveChart_levels'."
        )
    )

    args = parser.parse_args()

    csv_path = args.csv
    out_dir = args.output_dir

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # choix de la colonne énergie : energy_keV (catalogue IAEA) ou energy_kev (extras)
    if "energy_kev" in df.columns and "energy_keV" not in df.columns:
        energy_col = "energy_kev"
    elif "energy_keV".lower() in df.columns:
        # au cas où l'écriture soit 'energy_keV' ou 'energy_kev' mélangée
        energy_col = "energy_kev" if "energy_kev" in df.columns else "energy_keV".lower()
    else:
        raise ValueError(
            "Le CSV ne contient pas de colonne d'énergie 'energy_keV' ou 'energy_kev'. "
            "Vérifie le fichier produit."
        )

    # jpi ou jp
    spin_col = None
    if "jpi" in df.columns:
        spin_col = "jpi"
    elif "jp" in df.columns:
        spin_col = "jp"
    else:
        raise ValueError(
            "Le CSV ne contient ni colonne 'jpi' ni 'jp' pour le spin."
        )

    # filtrage optionnel sur la source (si on veut uniquement IAEA)
    if args.only_iaea and "source" in df.columns:
        df = df[
            df["source"].isin(
                ["IAEA_LiveChart_ground", "IAEA_LiveChart_levels"]
            )
        ].copy()

    # On ne travaille que sur les Z > 0
    if "z" not in df.columns:
        raise ValueError("Le CSV doit contenir une colonne 'z' (numéro atomique).")
    df = df[df["z"] > 0].copy()

    # A doit exister (normalement présent dans le catalogue, sinon on le reconstruit)
    if "a" not in df.columns:
        if "n" not in df.columns:
            raise ValueError("Le CSV doit contenir 'a' ou (z,n) pour reconstruire A.")
        df["a"] = df["z"].astype(int) + df["n"].astype(int)

    # symbole
    if "symbol" not in df.columns:
        df["symbol"] = "Z" + df["z"].astype(int).astype(str)

    # Calcul du spin signé
    df["spin_signed"] = df[spin_col].apply(parse_spin_jp)

    # Groupement par élément Z
    mask_z = (df["z"] >= args.min_z) & (df["z"] <= args.max_z)
    df = df[mask_z].copy()

    if df.empty:
        print("Aucune donnée dans l'intervalle de Z demandé.")
        return

    n_elements = 0
    n_plots = 0
    for z_val, group in df.groupby("z"):
        created = make_plots_for_element(group, out_dir=out_dir, energy_col=energy_col)
        if created:
            n_elements += 1
            # au moins un 2D; 3D peut manquer si pas d'énergie, mais on compte 2 "cibles"
            n_plots += 2

    print()
    print(f"Terminé : {n_elements} éléments traités (cible 2 graphes/élément).")
    print(f"Les graphes se trouvent dans :")
    print(f"  {os.path.join(out_dir, '2D')}")
    print(f"  {os.path.join(out_dir, '3D')}")


if __name__ == "__main__":
    main()
