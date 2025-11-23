# Nuclear Isotope Levels Toolkit

Outils Python pour :

- construire un catalogue unifié des niveaux nucléaires (ground + niveaux excités) à partir de l’API IAEA LiveChart (et hooks pour d’autres sources),
- générer des graphiques 2D/3D des spins et niveaux d’énergie par élément et par isotope (avec gestion des isomères m1/m2/m3).
- ces outils peuvent etre librement améliorés , c'est encore loin des graphs fait main en 2D pour les chirps m1,m2,m3 

---

## 1. Prérequis

- Python 3.9+ (recommandé 3.10+)
- `pip` ou `conda` pour installer les dépendances

### Dépendances principales


pip install requests pandas matplotlib scipy


## 2. build_nuclear_levels_catalog.py

- Construit un catalogue unifié des niveaux nucléaires (ground + niveaux excités) à partir de l’API IAEA LiveChart, avec hooks pour d’autres sources (NuDat, LNHB, KAERI).

### Fonctionnalités

Récupère les états fondamentaux (ground states) via LiveChart :

- soit directement avec fields=ground_states&nuclides=all,
- soit à partir d’un ground_states.csv déjà téléchargé.
- Récupère les niveaux excités pour chaque nucléide via fields=levels&nuclides=AZ (A+symbole).

Construit un catalogue “long” avec une ligne par niveau :

- Colonnes principales (IAEA) :

- source : IAEA_LiveChart_ground ou IAEA_LiveChart_levels
- z, n, a, symbol, nuclide
- energy_keV, energy_mev
- jpi (spin-parité)
- half_life_sec
- is_ground (booléen)
- extra_iaea_* (colonnes brutes supplémentaires)

Hooks prévus pour :

- un export NuDat au format JSON / CSV,
- des exports LNHB / KAERI déjà remis en DataFrame.

- Usage simple IAEA uniquement :

python build_nuclear_levels_catalog.py \
    --output nuclear_levels_catalog.csv

- ou en réutilisant un ground_states local :

python build_nuclear_levels_catalog.py \
    --iaea-ground-states ground_states.csv \
    --output nuclear_levels_catalog.csv

Options utiles

- Limiter l’énergie des niveaux excités :

python build_nuclear_levels_catalog.py \
    --iaea-ground-states ground_states.csv \
    --max-energy-kev 5000 \
    --output nuclear_levels_catalog_0_5MeV.csv

- Limiter le nombre de nucléides pour test :

python build_nuclear_levels_catalog.py \
    --max-nuclides 50 \
    --output nuclear_levels_catalog_small.csv

- Ajouter un export NuDat (hook générique à adapter au JSON ) :

python build_nuclear_levels_catalog.py \
    --iaea-ground-states ground_states.csv \
    --nudat-json nudat_export.json \
    --output nuclear_levels_catalog_with_nudat.csv


## 3. plot_spins_levels_from_catalog.py (preversion avec affichage simple non différencié entre isotopes et etats excités pour la 3D)

python plot_spins_levels_from_catalog.py \
    nuclear_levels_catalog.csv \
    -o plots_levels

- les options sont disponibles dans le header script


## 4. plot_spins_levels_from_catalog_params.py

Génère des graphes 2D et 3D par élément à partir du catalogue unifié (nuclear_levels_catalog.csv).

### Entrée attendue

Le CSV doit contenir (au minimum) :

- z, a, symbol
- energy_kev (ou energy_keV avant passage en minuscule)
- jpi (ou jp)
- is_ground
- éventuellement une colonne identifiant les isomères (m1, m2, m3) pour un traitement spécial (ex. state_label).

Graphique 2D (par élément Z)

- Axe x : numéro de masse A
- Axe y : spin signé J (extraction de Jπ → ±J)

Composants :

- courbe lissée (spline cubique si SciPy dispo) passant par les spins des états fondamentaux (is_ground=True),
- points pour tous les niveaux (ground + excités),
- les isomères m1/m2/m3 sont légèrement décalés en abscisse à droite du A du ground correspondant (A, A+m1, A+m2, A+m3 sur une même échelle globale).

Graphique 3D (par élément Z)

- Axe x : A
- Axe y : energy_kev
- Axe z : spin signé J

Composants :

- points ground, isomères m* et autres niveaux avec 3 couleurs distinctes :
- ground : bleu (C0)
- m1/m2/m3 : orange (C1, marqueur ^)
- autres niveaux : vert (C2, .)

trois courbes reliants :

- tous les points m1 ensemble,
- tous les points m2 ensemble,
- tous les points m3 ensemble,
- (triés par A / énergie).

Usage standard

python plot_spins_levels_from_catalog_params.py \
    nuclear_levels_catalog.csv \
    -o plots_levels

Les figures générées :

 - plots_levels/2D/Z046_Pd_spins_levels_2D.png
 - plots_levels/3D/Z046_Pd_spins_levels_3D.png
 - etc. pour chaque Z dans l’intervalle demandé.

Restreindre à une plage en Z :

 python plot_spins_levels_from_catalog_params.py nuclear_levels_catalog.csv \
    -o plots_levels_Pd \
    --min-z 46 --max-z 46

Utiliser uniquement les données IAEA :

python plot_spins_levels_from_catalog_params.py nuclear_levels_catalog.csv \
    -o plots_levels_iaea_only \
    --only-iaea

Filtrer par énergie (en keV), mais en gardant toujours les ground + m1/m2/m3 :

python plot_spins_levels_from_catalog_params.py nuclear_levels_catalog.csv \
    -o plots_levels_0_5MeV \
    --energy-max-kev 5000

Réduire le nombre de niveaux excités “extras” par isotope (Z,A) :

python plot_spins_levels_from_catalog_params.py nuclear_levels_catalog.csv \
    -o plots_levels_light \
    --energy-max-kev 5000 \
    --max-extras-per-isotope 15

Activer le traitement spécial des isomères (m1/m2/m3) :

python plot_spins_levels_from_catalog_params.py nuclear_levels_catalog.csv \
    -o plots_levels_isomers \
    --energy-max-kev 5000 \
    --max-extras-per-isotope 10 \
    --isomer-col state_label \
    --isomer-values m1,m2,m3

dans ce cas :

--isomer-col doit pointer vers une colonne du CSV contenant des labels de niveaux (par ex. m1, m2, m3).
--isomer-values est la liste (séparée par virgules) des valeurs à considérer comme isomères.


## 5.  Notes / Limitations

### L’API IAEA LiveChart ne doit pas être sur-sollicitée :

- utiliser un délai --delay approprié dans build_nuclear_levels_catalog.py si tu lances des scans complets.
- Les hooks NuDat / LNHB / KAERI dans build_nuclear_levels_catalog.py sont génériques et doivent être adaptés au format exact des exports obtenus.
- La détection et le traitement des isomères m1/m2/m3 supposent :
- une colonne identifiant les niveaux métastables,
- des labels cohérents (m1, m2, m3 ou équivalents).
