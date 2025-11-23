#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spins_levels_from_catalog_params.py

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
       * les isomères m1/m2/m3 légèrement décalés en abscisse à droite
         de leur A_gs (A, A_m1, A_m2, A_m3).

  2) Graphique 3D (par élément) :
       - x = numéro de masse A
       - y = énergie du niveau (keV)
       - z = spin signé J (Jπ → ±J)
     vue en légère perspective, avec :
       * points ground, isomères m* et autres niveaux de couleurs distinctes
       * courbe qui relie tous les m1, tous les m2, tous les m3.

Filtres :
  - --energy-min-kev / --energy-max-kev :
        filtrage par plage d'énergie (keV),
        MAIS les niveaux is_ground et les isomères m1/m2/m3 sont
        toujours conservés.
  - --isomer-col / --isomer-values :
        permettent de marquer les isomères (m1/m2/m3) dans le CSV.
  - --max-extras-per-isotope :
        pour chaque (Z,A), ne garder que les ground + isomères
        + au plus N autres niveaux (choisis par énergie croissante).

Le CSV d'entrée doit contenir (au minimum) les colonnes :
  - z, a, symbol
  - energy_kev  (ou energy_keV avant passage en lower)
  - jpi (ou jp si jpi absent)
  - is_ground (booléen ou équivalent, si possible)

Dépendances :
    pip install numpy pandas matplotlib
    pip install scipy        # pour l'interpolation spline (recommandé)

Usage :
    python plot_spins_levels_from_catalog_params.py \
        nuclear_levels_catalog.csv \
        -o plots_levels

Exemples :
    # limiter aux niveaux < 5000 keV, garder m1/m2/m3 et ground
    python plot_spins_levels_from_catalog_params.py nuclear_levels_catalog.csv \
        -o plots_levels_0_5MeV \
        --energy-max-kev 5000

    # + limiter le nombre d'extras par isotope
    python plot_spins_levels_from_catalog_params.py nuclear_levels_catalog.csv \
        -o plots_levels_light \
        --energy-max-kev 5000 \
        --max-extras-per-isotope 15

    # avec colonne des isomères explicite (ex: 'state_label' contenant m1/m2/m3)
    python plot_spins_levels_from_catalog_params.py nuclear_levels_catalog.csv \
        -o plots_levels_isomers \
        --energy-max-kev 5000 \
        --max-extras-per-isotope 10 \
        --isomer-col state_label \
        --isomer-values m1,m2,m3
"""

import argparse
import math
import os
import re
from typing import Optional, List

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
# Globals pour la gestion des isomères
# ---------------------------------------------------------------------------

ISOMER_COL_NAME: Optional[str] = None      # ex: "state_label"
ISOMER_LABELS: List[str] = []              # ex: ["m1", "m2", "m3"] (lowercase)


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

    group : DataFrame pour un Z donné, avec :
        'z', 'a', 'symbol', 'spin_signed', 'is_ground', 'is_isomer', energy_col.
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

    # ------------------------------------------------------------------
    # 2D : spins vs A (avec interpolation sur les ground states)
    # ------------------------------------------------------------------

    # Ground states (is_ground = True ou équivalent)
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

    # Tous les niveaux (pour les points) + abscisses décalées pour m1/m2/m3
    sub_2d = sub.sort_values(["a", energy_col]).copy()
    sub_2d["x_2d"] = sub_2d["a"].astype(float)

    # Décalage des isomères m1/m2/m3 sur l'axe des A
    # A_gs      -> x = A
    # A_m1      -> x = A + delta1
    # A_m2      -> x = A + delta2
    # A_m3      -> x = A + delta3
    if ISOMER_COL_NAME and ISOMER_COL_NAME in sub_2d.columns and ISOMER_LABELS:
        col = sub_2d[ISOMER_COL_NAME].astype(str).str.lower()
        # petits offsets, à adapter si besoin
        base_step = 0.15
        offsets = {
            lab: base_step * (i + 1)
            for i, lab in enumerate(ISOMER_LABELS)
        }
        for lab, off in offsets.items():
            mask_lab = col == lab
            sub_2d.loc[mask_lab, "x_2d"] = sub_2d.loc[mask_lab, "x_2d"] + off

    # Séparation ground / excités pour les points
    if "is_ground" in sub_2d.columns:
        excited = sub_2d[~sub_2d["is_ground"].astype(bool)]
    else:
        excited = pd.DataFrame()

    fig2d, ax2d = plt.subplots(figsize=(7, 4))

    # Courbe lissée (ou non) ground
    ax2d.plot(
        x_dense,
        z_dense,
        linestyle="-",
        label="ground (interp.)" if use_spline else "ground"
    )

    # Points ground (à A exact)
    ax2d.plot(xs_g, zs_g, marker="o", linestyle="", label="ground (points)")

    # Points niveaux excités (dont m*, déjà décalés en x_2d pour m1/m2/m3)
    if not excited.empty:
        ax2d.plot(
            excited["x_2d"].values.astype(float),
            excited["spin_signed"].values.astype(float),
            linestyle="",
            marker="x",
            label="niveaux excités (incl. m*)"
        )

    ax2d.set_xlabel("Numéro de masse A (m* légèrement décalés)")
    ax2d.set_ylabel("Spin signé J (Jπ → ±J)")
    ax2d.set_title(f"{symbol} (Z={z_val}) – spins par isotopes (ground + niveaux)")
    ax2d.axhline(0.0, linewidth=0.8)
    ax2d.legend(loc="best")
    fig2d.tight_layout()

    out_path_2d = os.path.join(out_2d, f"{prefix}_spins_levels_2D.png")
    fig2d.savefig(out_path_2d, dpi=150)
    plt.close(fig2d)

    # ------------------------------------------------------------------
    # 3D : A vs énergie vs spin (avec couleurs distinctes, + courbes m1/m2/m3)
    # ------------------------------------------------------------------

    # On garde seulement les lignes avec énergie définie
    sub_3d = sub[sub[energy_col].notna()].copy()
    if sub_3d.empty:
        print(f"[Z={z_val:3d}] {symbol:>3} : pas d'énergie définie pour les niveaux, graphe 3D ignoré.")
        return True  # 2D a été généré

    # Masques : ground / isomères m* / autres excités
    g_mask = sub_3d["is_ground"].astype(bool) if "is_ground" in sub_3d.columns else (sub_3d["a"] < 0)
    iso_mask = sub_3d["is_isomer"].astype(bool) if "is_isomer" in sub_3d.columns else (sub_3d["a"] < 0)
    other_mask = ~(g_mask | iso_mask)

    fig3d = plt.figure(figsize=(7, 4))
    ax3d = fig3d.add_subplot(111, projection="3d")

    # Ground states : couleur 1
    g_points = sub_3d[g_mask]
    if not g_points.empty:
        ax3d.scatter(
            g_points["a"].values.astype(float),
            g_points[energy_col].values.astype(float),
            g_points["spin_signed"].values.astype(float),
            marker="o",
            color="C0",
            label="ground"
        )

    # Isomères m1/m2/m3 : couleur 2
    iso_points = sub_3d[iso_mask]
    if not iso_points.empty:
        # Points isolés
        ax3d.scatter(
            iso_points["a"].values.astype(float),
            iso_points[energy_col].values.astype(float),
            iso_points["spin_signed"].values.astype(float),
            marker="^",
            color="C1",
            label="m* (isomères)"
        )
        # Courbes m1 / m2 / m3 (si infos disponibles)
        if ISOMER_COL_NAME and ISOMER_COL_NAME in iso_points.columns and ISOMER_LABELS:
            col_iso = iso_points[ISOMER_COL_NAME].astype(str).str.lower()
            for lab in ISOMER_LABELS:
                mask_lab = col_iso == lab
                pts_lab = iso_points[mask_lab]
                if pts_lab.empty:
                    continue
                xs_lab = pts_lab["a"].values.astype(float)
                ys_lab = pts_lab[energy_col].values.astype(float)
                zs_lab = pts_lab["spin_signed"].values.astype(float)
                # tri par A croissant (éventuellement par énergie secondaire)
                order = np.lexsort((ys_lab, xs_lab))
                xs_lab = xs_lab[order]
                ys_lab = ys_lab[order]
                zs_lab = zs_lab[order]
                ax3d.plot(
                    xs_lab,
                    ys_lab,
                    zs_lab,
                    color="C1",
                    linestyle="-",
                    linewidth=1.0,
                )

    # Autres niveaux excités : couleur 3
    other_points = sub_3d[other_mask]
    if not other_points.empty:
        ax3d.scatter(
            other_points["a"].values.astype(float),
            other_points[energy_col].values.astype(float),
            other_points["spin_signed"].values.astype(float),
            marker=".",
            color="C2",
            label="niveaux excités"
        )

    ax3d.set_xlabel("Numéro de masse A")
    ax3d.set_ylabel(f"Énergie niveau ({energy_col})")
    ax3d.set_zlabel("Spin signé J (Jπ → ±J)")
    ax3d.set_title(f"{symbol} (Z={z_val}) – niveaux (A, énergie, spin)")

    # Légère perspective + légende
    ax3d.view_init(elev=25, azim=-60)
    ax3d.legend(loc="best")

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
    global ISOMER_COL_NAME, ISOMER_LABELS

    parser = argparse.ArgumentParser(
        description=(
            "Génère, à partir d'un catalogue CSV (ground + niveaux + extras), "
            "des graphes 2D et 3D des spins/énergies par élément, "
            "avec filtres d'énergie, traitement m1/m2/m3 et réduction des extras."
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
    # --- Filtres d'énergie ---
    parser.add_argument(
        "--energy-min-kev",
        type=float,
        default=None,
        help="Énergie minimale des niveaux (keV)."
    )
    parser.add_argument(
        "--energy-max-kev",
        type=float,
        default=None,
        help="Énergie maximale des niveaux (keV)."
    )
    # --- Identification des isomères (m1/m2/m3) ---
    parser.add_argument(
        "--isomer-col",
        default=None,
        help=(
            "Nom de la colonne qui indique les états métastables "
            "(m1/m2/m3). Aucun traitement spécial si absent."
        )
    )
    parser.add_argument(
        "--isomer-values",
        default="m1,m2,m3",
        help=(
            "Valeurs dans isomer-col qui désignent les isomères, séparées "
            "par des virgules (défaut: 'm1,m2,m3')."
        )
    )
    # --- Réduction des extras ---
    parser.add_argument(
        "--max-extras-per-isotope",
        type=int,
        default=None,
        help=(
            "Nombre maximum de niveaux excités supplémentaires (hors ground "
            "et isomères) à garder par isotope (Z,A). "
            "Les plus bas en énergie sont conservés."
        )
    )

    args = parser.parse_args()

    csv_path = args.csv
    out_dir = args.output_dir

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Colonne énergie : après lower, 'energy_keV' → 'energy_kev'
    if "energy_kev" in df.columns:
        energy_col = "energy_kev"
    elif "energy" in df.columns:
        energy_col = "energy"
    else:
        raise ValueError(
            "Le CSV ne contient pas de colonne d'énergie 'energy_kev' ou 'energy'. "
            "Vérifie le fichier produit."
        )

    # jpi ou jp
    if "jpi" in df.columns:
        spin_col = "jpi"
    elif "jp" in df.columns:
        spin_col = "jp"
    else:
        raise ValueError(
            "Le CSV ne contient ni colonne 'jpi' ni 'jp' pour le spin."
        )

    # Gestion globale de la colonne des isomères et des labels
    if args.isomer_col:
        ISOMER_COL_NAME = args.isomer_col.lower()
        ISOMER_LABELS = [
            v.strip().lower()
            for v in args.isomer_values.split(",")
            if v.strip()
        ]
    else:
        ISOMER_COL_NAME = None
        ISOMER_LABELS = []

    # filtrage optionnel sur la source (si on veut uniquement IAEA)
    if args.only_iaea and "source" in df.columns:
        df = df[
            df["source"].isin(
                ["iaea_livechart_ground", "iaea_livechart_levels"]
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

    # Colonne is_ground : par défaut False si absente
    if "is_ground" not in df.columns:
        df["is_ground"] = False

    # Colonne is_isomer (m1/m2/m3) suivant les options
    df["is_isomer"] = False
    if ISOMER_COL_NAME and ISOMER_COL_NAME in df.columns and ISOMER_LABELS:
        col_iso = df[ISOMER_COL_NAME].astype(str).str.lower()
        df["is_isomer"] = col_iso.isin(ISOMER_LABELS)

    # --- Filtrage par énergie, en conservant toujours ground + isomères ---
    if args.energy_min_kev is not None or args.energy_max_kev is not None:
        mask_energy = pd.Series(True, index=df.index)
        if args.energy_min_kev is not None:
            mask_energy &= df[energy_col].notna()
            mask_energy &= df[energy_col] >= args.energy_min_kev
        if args.energy_max_kev is not None:
            mask_energy &= df[energy_col].notna()
            mask_energy &= df[energy_col] <= args.energy_max_kev

        always_keep = df["is_ground"].astype(bool) | df["is_isomer"].astype(bool)
        df = df[mask_energy | always_keep].copy()

    # --- Réduction des extras par isotope (Z,A) ---
    if args.max_extras_per_isotope is not None and args.max_extras_per_isotope >= 0:
        max_extra = args.max_extras_per_isotope

        def downsample_group(g: pd.DataFrame) -> pd.DataFrame:
            """
            Pour un groupe (Z,A), on garde :
              - tous les ground (is_ground),
              - tous les isomères (is_isomer),
              - au plus max_extra autres niveaux, choisis par énergie croissante.
            """
            g = g.copy()
            mask_ground = g["is_ground"].astype(bool)
            mask_iso = g["is_isomer"].astype(bool)
            mask_extra = ~(mask_ground | mask_iso)

            keep = g[mask_ground | mask_iso]
            extras = g[mask_extra]

            if len(extras) <= max_extra:
                return g

            # On prend les extras les plus bas en énergie
            extras = extras.sort_values(energy_col).head(max_extra)
            return pd.concat([keep, extras], ignore_index=False)

        df = df.groupby(["z", "a"], group_keys=False).apply(downsample_group)

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
            # on vise 2 graphes/élément (2D + 3D)
            n_plots += 2

    print()
    print(f"Terminé : {n_elements} éléments traités (cible 2 graphes/élément).")
    print(f"Les graphes se trouvent dans :")
    print(f"  {os.path.join(out_dir, '2D')}")
    print(f"  {os.path.join(out_dir, '3D')}")


if __name__ == "__main__":
    main()
