#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_nuclear_levels_catalog.py

But :
    - Interroger l’API REST IAEA LiveChart pour :
        * les états fondamentaux (ground_states)
        * les niveaux excités (levels)
      pour l’ensemble des noyaux disponibles.
    - Fusionner tout ça dans un CSV unifié :
        Z, N, A, symbole, nuclide (ex. 11Be), énergie de niveau,
        Jπ, demi-vie, type de niveau (ground/excited), source, etc.
    - Prévoir des hooks pour ajouter plus tard :
        * des exports NuDat 3 (CSV/JSON)     -> source = "NuDat"
        * des exports LNHB / LARA / KAERI    -> source = "LNHB", "KAERI", etc.

Remarques :
    - L’API LiveChart utilisée est :
        base : https://nds.iaea.org/relnsd/v0/data
      avec les paramètres :
        fields=ground_states&nuclides=all
        fields=levels&nuclides=11Be   (exemple)  [AZ = A + symbole]
      (cf. code d’exemple public : Nuclear Physics 101, 2022). ¹

    - Pour NuDat 3 :
        l’interface web permet d’exporter CSV/JSON (chart complet),
        mais la Web API REST publique n’est pas encore standardisée.
        On lit donc ces exports localement si disponibles. ²

Dépendances :
    pip install requests pandas

Usage typique :
    # 1) tout récupérer depuis LiveChart et produire iaea_levels_catalog.csv
    python build_nuclear_levels_catalog.py \
        --output iaea_levels_catalog.csv

    # 2) idem, mais en réutilisant un ground_states.csv déjà téléchargé
    python build_nuclear_levels_catalog.py \
        --iaea-ground-states ground_states.csv \
        --output iaea_levels_catalog.csv

    # 3) ajouter un export NuDat JSON en entrée (déjà téléchargé via l’UI NuDat)
    python build_nuclear_levels_catalog.py \
        --nudat-json nudat_export.json \
        --output levels_catalog_with_nudat.csv
"""

import argparse
import io
import os
import sys
import time
from typing import List, Optional

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration par défaut
# ---------------------------------------------------------------------------

# Base officielle LiveChart (cf. doc API v0)
LIVECHART_BASE_URL = "https://nds.iaea.org/relnsd/v0/data"

DEFAULT_USER_AGENT = (
    "NuclearLevelsAggregator/0.1 "
    "(personal research script; please contact maintainer before heavy use)"
)


# ---------------------------------------------------------------------------
# Utilitaires de lecture LiveChart
# ---------------------------------------------------------------------------

def make_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    """
    Crée une session HTTP avec un User-Agent explicite
    (évite les 403/filtrages côté serveur).
    """
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent})
    return sess


def livechart_read_csv(
    session: requests.Session,
    fields: str,
    nuclides: str,
    base_url: str = LIVECHART_BASE_URL,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """
    Interroge l’API LiveChart pour un couple (fields, nuclides),
    renvoie un DataFrame pandas.

    Exemple :
        livechart_read_csv(sess, "ground_states", "all")
        livechart_read_csv(sess, "levels", "11Be")
    """
    params = {
        "fields": fields,
        "nuclides": nuclides,
    }
    resp = session.get(base_url, params=params, timeout=timeout)
    resp.raise_for_status()

    # LiveChart renvoie du CSV texte
    content = io.StringIO(resp.text)
    df = pd.read_csv(content)

    # normalisation des noms de colonnes en minuscules pour la suite
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Chargement des ground states IAEA
# ---------------------------------------------------------------------------

def load_iaea_ground_states(
    session: requests.Session,
    ground_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Charge les états fondamentaux IAEA (une ligne par nuclide).

    - Si ground_csv_path est fourni : lit ce fichier local.
    - Sinon : interroge directement LiveChart (fields=ground_states&nuclides=all).

    La table retournée contient typiquement :
        z, n, symbol, binding, half_life_sec, jp, ...
    """
    if ground_csv_path is not None:
        if not os.path.isfile(ground_csv_path):
            raise FileNotFoundError(
                f"Fichier ground_states introuvable : {ground_csv_path}"
            )
        df = pd.read_csv(ground_csv_path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    # Appel à LiveChart
    print("[INFO] Récupération IAEA LiveChart : ground_states (tous les nuclides)...")
    df = livechart_read_csv(session, fields="ground_states", nuclides="all")
    print(f"[INFO]   -> {len(df)} lignes (nuclides)")
    return df


def build_nuclide_code(a: int, symbol: str) -> str:
    """
    Construit le code AZ utilisé par LiveChart à partir de A et du symbole chimique.
    Exemple : A=11, symbol="Be" -> "11Be"
    """
    return f"{int(a)}{str(symbol).strip()}"


# ---------------------------------------------------------------------------
# Chargement des niveaux excités IAEA
# ---------------------------------------------------------------------------

def fetch_levels_for_nuclide(
    session: requests.Session,
    nuclide_code: str,
    base_url: str = LIVECHART_BASE_URL,
    max_energy_keV: Optional[float] = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """
    Récupère les niveaux excités pour un nuclide donné (ex. "11Be").

    - Utilise fields=levels&nuclides=nuclide_code
    - Filtre éventuellement sur max_energy_keV (si fourni)
    - Ajoute une colonne 'nuclide' avec la chaîne AZ.

    Si aucun niveau n’est trouvé, renvoie un DataFrame vide.
    """
    try:
        df = livechart_read_csv(
            session=session,
            fields="levels",
            nuclides=nuclide_code,
            base_url=base_url,
            timeout=timeout,
        )
    except Exception as e:
        print(f"[WARN] Échec récupération niveaux pour {nuclide_code}: {e}", file=sys.stderr)
        return pd.DataFrame()

    if df.empty:
        return df

    # On s’assure que la colonne énergie existe
    if "energy" not in df.columns:
        return pd.DataFrame()

    # Filtre éventuel par énergie
    if max_energy_keV is not None:
        df = df[df["energy"] <= max_energy_keV]

    if df.empty:
        return df

    df["nuclide"] = nuclide_code
    return df


def fetch_all_levels(
    session: requests.Session,
    ground_df: pd.DataFrame,
    base_url: str = LIVECHART_BASE_URL,
    max_energy_keV: Optional[float] = None,
    delay: float = 0.1,
    max_nuclides: Optional[int] = None,
) -> pd.DataFrame:
    """
    Balaye tous les nuclides connus dans ground_df, et appelle
    l’API LiveChart pour récupérer leurs niveaux excités.

    Paramètres :
        - max_energy_keV : optionnel, limite sur l’énergie des niveaux.
        - delay : pause entre deux appels HTTP (politesse vis-à-vis du serveur).
        - max_nuclides : limite le nombre de nuclides à traiter (debug/tests).

    Retour :
        DataFrame avec toutes les lignes "levels" de LiveChart,
        enrichies avec Z, N, A, symbol.
    """
    ground = ground_df.copy()
    # normalisation des noms de colonnes
    ground.columns = [str(c).strip().lower() for c in ground.columns]

    if not {"z", "n", "symbol"}.issubset(set(ground.columns)):
        raise ValueError(
            "ground_states doit contenir au moins les colonnes 'z', 'n', 'symbol'"
        )

    # on calcule A = Z + N
    ground["a"] = ground["z"].astype(int) + ground["n"].astype(int)

    all_levels: List[pd.DataFrame] = []
    total_nuclides = len(ground)
    print(f"[INFO] Récupération des niveaux excités pour {total_nuclides} nuclides...")

    for idx, row in ground.iterrows():
        z = int(row["z"])
        n = int(row["n"])
        a = int(row["a"])
        symbol = str(row["symbol"]).strip()
        nuclide_code = build_nuclide_code(a, symbol)

        print(
            f"[INFO]  [{idx+1}/{total_nuclides}] Nuclide {nuclide_code} "
            f"(Z={z}, N={n})..."
        )

        df_levels = fetch_levels_for_nuclide(
            session=session,
            nuclide_code=nuclide_code,
            base_url=base_url,
            max_energy_keV=max_energy_keV,
        )

        if not df_levels.empty:
            # On ajoute les infos de base
            df_levels["z"] = z
            df_levels["n"] = n
            df_levels["a"] = a
            df_levels["symbol"] = symbol
            all_levels.append(df_levels)
            print(f"[INFO]      -> {len(df_levels)} niveaux ajoutés.")
        else:
            print(f"[INFO]      -> Aucun niveau trouvé.")

        if max_nuclides is not None and len(all_levels) >= max_nuclides:
            print(
                f"[INFO] Limite max_nuclides atteinte ({max_nuclides}). "
                "Arrêt anticipé."
            )
            break

        if delay > 0:
            time.sleep(delay)

    if not all_levels:
        print("[WARN] Aucun niveau excité trouvé pour l’ensemble des nuclides.")
        return pd.DataFrame()

    levels_all = pd.concat(all_levels, ignore_index=True)
    print(f"[INFO] Total niveaux excités IAEA LiveChart : {len(levels_all)} lignes.")
    return levels_all


# ---------------------------------------------------------------------------
# Fusion en un catalogue unifié
# ---------------------------------------------------------------------------

def build_unified_catalog(
    ground_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    nudat_df: Optional[pd.DataFrame] = None,
    lnhb_df: Optional[pd.DataFrame] = None,
    kaeri_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construit un catalogue unifié des niveaux :

        - Ground states IAEA -> source = 'IAEA_LiveChart_ground'
        - Levels IAEA        -> source = 'IAEA_LiveChart_levels'
        - + éventuellement données NuDat / LNHB / KAERI fusionnées.

    Le catalog final est "long" : une ligne = un niveau.

    Colonnes principales :
        source, z, n, a, symbol, nuclide,
        energy_keV, energy_mev, jpi, half_life_sec,
        is_ground, extra_* (colonnes brutes des sources).

    Pour NuDat / LNHB / KAERI, on suppose que les DataFrames fournis
    ont déjà été normalisés en quelque chose de cohérent (hooks).
    """

    # --- 1) Ground states IAEA -> niveaux "synthétiques" à E=0
    g = ground_df.copy()
    g.columns = [str(c).strip().lower() for c in g.columns]

    if not {"z", "n", "symbol"}.issubset(set(g.columns)):
        raise ValueError(
            "ground_df doit contenir au moins les colonnes 'z', 'n', 'symbol'"
        )

    g["z"] = g["z"].astype(int)
    g["n"] = g["n"].astype(int)
    g["a"] = g["z"] + g["n"]
    g["nuclide"] = g.apply(
        lambda row: build_nuclide_code(int(row["a"]), str(row["symbol"])), axis=1
    )

    # Jπ et demi-vie peuvent être absents ou NaN
    jpi_col = "jp" if "jp" in g.columns else None
    hl_col = "half_life_sec" if "half_life_sec" in g.columns else None

    ground_cat = pd.DataFrame(
        {
            "source": "IAEA_LiveChart_ground",
            "z": g["z"],
            "n": g["n"],
            "a": g["a"],
            "symbol": g["symbol"],
            "nuclide": g["nuclide"],
            # ground state -> énergie nulle par définition
            "energy_keV": 0.0,
            "energy_mev": 0.0,
            "jpi": g[jpi_col] if jpi_col else None,
            "half_life_sec": g[hl_col] if hl_col else None,
            "is_ground": True,
        }
    )

    # Garder en mémoire quelques colonnes brutes utiles (prefix extra_)
    for col in g.columns:
        if col in {"z", "n", "a", "symbol", "nuclide", jpi_col, hl_col}:
            continue
        ground_cat[f"extra_iaea_ground_{col}"] = g[col]

    # --- 2) Levels IAEA -> niveaux excités
    levels_cat_list: List[pd.DataFrame] = []
    if levels_df is not None and not levels_df.empty:
        lv = levels_df.copy()
        lv.columns = [str(c).strip().lower() for c in lv.columns]

        # s'assurer qu’on a bien z,n,a,symbol,nuclide
        if not {"z", "n", "symbol"}.issubset(set(lv.columns)):
            raise ValueError(
                "levels_df doit contenir au moins les colonnes 'z', 'n', 'symbol'"
            )
        if "a" not in lv.columns:
            lv["a"] = lv["z"].astype(int) + lv["n"].astype(int)
        if "nuclide" not in lv.columns:
            lv["nuclide"] = lv.apply(
                lambda row: build_nuclide_code(
                    int(row["a"]), str(row["symbol"])
                ),
                axis=1,
            )

        # energy en keV (cf. scripts publics LiveChart)
        if "energy" not in lv.columns:
            raise ValueError(
                "levels_df (IAEA) doit contenir une colonne 'energy' (keV)."
            )

        # jpi / half-life (si dispo)
        jpi_col_lv = "jp" if "jp" in lv.columns else None
        hl_col_lv = "half_life_sec" if "half_life_sec" in lv.columns else None

        levels_cat = pd.DataFrame(
            {
                "source": "IAEA_LiveChart_levels",
                "z": lv["z"].astype(int),
                "n": lv["n"].astype(int),
                "a": lv["a"].astype(int),
                "symbol": lv["symbol"],
                "nuclide": lv["nuclide"],
                "energy_keV": lv["energy"].astype(float),
                "energy_mev": lv["energy"].astype(float) / 1000.0,
                "jpi": lv[jpi_col_lv] if jpi_col_lv else None,
                "half_life_sec": lv[hl_col_lv] if hl_col_lv else None,
                "is_ground": False,
            }
        )

        # Colonnes brutes restantes
        for col in lv.columns:
            if col in {
                "z", "n", "a", "symbol", "nuclide", "energy",
                jpi_col_lv, hl_col_lv
            }:
                continue
            levels_cat[f"extra_iaea_levels_{col}"] = lv[col]

        levels_cat_list.append(levels_cat)

    # --- 3) Hooks pour NuDat / LNHB / KAERI (si déjà normalisés en DataFrame)

    extra_sources: List[pd.DataFrame] = []

    if nudat_df is not None and not nudat_df.empty:
        df = nudat_df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        # ICI : à adapter selon le format concret de ton export NuDat (JSON/CSV)…
        # Pour l’instant on laisse un canevas minimal :
        if not {"z", "n", "a", "symbol", "energy_kev"}.issubset(df.columns):
            print(
                "[WARN] nuDat_df ne contient pas les colonnes attendues "
                "('z','n','a','symbol','energy_keV'). Hook à adapter.",
                file=sys.stderr,
            )
        else:
            df["source"] = "NuDat_export"
            df["energy_mev"] = df["energy_kev"].astype(float) / 1000.0
            df["is_ground"] = df.get("is_ground", False)
            extra_sources.append(df)

    if lnhb_df is not None and not lnhb_df.empty:
        df = lnhb_df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        # Idem : adapter selon ton format LNHB/LARA
        if not {"z", "n", "a", "symbol", "energy_kev"}.issubset(df.columns):
            print(
                "[WARN] lnhb_df ne contient pas les colonnes attendues "
                "('z','n','a','symbol','energy_keV'). Hook à adapter.",
                file=sys.stderr,
            )
        else:
            df["source"] = "LNHB_export"
            df["energy_mev"] = df["energy_kev"].astype(float) / 1000.0
            df["is_ground"] = df.get("is_ground", False)
            extra_sources.append(df)

    if kaeri_df is not None and not kaeri_df.empty:
        df = kaeri_df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"z", "n", "a", "symbol", "energy_kev"}.issubset(df.columns):
            print(
                "[WARN] kaeri_df ne contient pas les colonnes attendues "
                "('z','n','a','symbol','energy_keV'). Hook à adapter.",
                file=sys.stderr,
            )
        else:
            df["source"] = "KAERI_export"
            df["energy_mev"] = df["energy_kev"].astype(float) / 1000.0
            df["is_ground"] = df.get("is_ground", False)
            extra_sources.append(df)

    # --- 4) Concaténation globale
    catalog_parts: List[pd.DataFrame] = [ground_cat]
    if levels_cat_list:
        catalog_parts.extend(levels_cat_list)
    if extra_sources:
        catalog_parts.extend(extra_sources)

    catalog = pd.concat(catalog_parts, ignore_index=True)

    # Tri simple : par Z, A, énergie
    catalog = catalog.sort_values(by=["z", "a", "energy_keV"], ignore_index=True)

    return catalog


# ---------------------------------------------------------------------------
# Hooks : lecture d’exports NuDat / autres (facultatif)
# ---------------------------------------------------------------------------

def load_nudat_json(path: str) -> pd.DataFrame:
    """
    Lecture d’un export NuDat 3 au format JSON (via l’interface NuDat).
    NOTE :
        Le format n’est pas standardisé ici, il faudra adapter cette fonction
        à la structure concrète du fichier output.json que tu téléchargeras.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fichier NuDat JSON introuvable : {path}")
    # Place-holder : à adapter lorsque tu auras un exemple réel
    print("[INFO] Lecture NuDat JSON (hook générique, à adapter).")
    df = pd.read_json(path)
    return df


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Construit un CSV unifié des niveaux d’énergie nucléaires "
            "à partir de l’API IAEA LiveChart (et éventuellement d’autres sources)."
        )
    )

    parser.add_argument(
        "--iaea-ground-states",
        help=(
            "Chemin vers un ground_states.csv déjà téléchargé "
            "(sinon LiveChart sera interrogée directement)."
        ),
    )
    parser.add_argument(
        "--output",
        default="nuclear_levels_catalog.csv",
        help="Nom du fichier CSV de sortie (défaut: nuclear_levels_catalog.csv).",
    )
    parser.add_argument(
        "--max-energy-kev",
        type=float,
        default=None,
        help="Énergie max des niveaux excités (keV) pour IAEA (optionnel).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Pause (en secondes) entre deux requêtes LiveChart (défaut: 0.1).",
    )
    parser.add_argument(
        "--max-nuclides",
        type=int,
        default=None,
        help="Limiter le nombre de nuclides (debug/tests).",
    )
    parser.add_argument(
        "--nudat-json",
        help=(
            "Chemin vers un export NuDat JSON (optionnel, à fusionner dans le CSV). "
            "La fonction load_nudat_json() est un hook à adapter."
        ),
    )
    # Hooks pour LNHB / KAERI : on les laisse en option simple (CSV pré-formatté)
    parser.add_argument(
        "--lnhb-csv",
        help="Export LNHB/LARA déjà mis en CSV local (optionnel).",
    )
    parser.add_argument(
        "--kaeri-csv",
        help="Export KAERI déjà mis en CSV local (optionnel).",
    )

    args = parser.parse_args()

    session = make_session()

    # 1) Ground states IAEA
    ground_df = load_iaea_ground_states(
        session=session, ground_csv_path=args.iaea_ground_states
    )

    # 2) Levels IAEA
    levels_df = fetch_all_levels(
        session=session,
        ground_df=ground_df,
        max_energy_keV=args.max_energy_kev,
        delay=args.delay,
        max_nuclides=args.max_nuclides,
    )

    # 3) Sources supplémentaires (facultatif)
    nudat_df = None
    if args.nudat_json:
        nudat_df = load_nudat_json(args.nudat_json)

    lnhb_df = None
    if args.lnhb_csv:
        if not os.path.isfile(args.lnhb_csv):
            raise FileNotFoundError(f"Fichier LNHB CSV introuvable : {args.lnhb_csv}")
        lnhb_df = pd.read_csv(args.lnhb_csv)

    kaeri_df = None
    if args.kaeri_csv:
        if not os.path.isfile(args.kaeri_csv):
            raise FileNotFoundError(f"Fichier KAERI CSV introuvable : {args.kaeri_csv}")
        kaeri_df = pd.read_csv(args.kaeri_csv)

    # 4) Construction du catalogue unifié
    catalog = build_unified_catalog(
        ground_df=ground_df,
        levels_df=levels_df,
        nudat_df=nudat_df,
        lnhb_df=lnhb_df,
        kaeri_df=kaeri_df,
    )

    # 5) Sauvegarde CSV
    output_path = args.output
    catalog.to_csv(output_path, index=False)
    print()
    print(f"[OK] Catalogue complet écrit dans : {output_path}")
    print(f"[INFO] Nombre total de lignes (ground + niveaux + extras) : {len(catalog)}")


if __name__ == "__main__":
    main()
