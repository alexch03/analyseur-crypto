[🇬🇧 English](RESEARCH_NOTES.md) | [🇫🇷 Français](RESEARCH_NOTES.fr.md)

# Plan d'expérimentation (DOE) — Analyseur crypto
**Pré-enregistré le 2026-05-30. Les hypothèses et critères ci-dessous sont figés AVANT de regarder le holdout.**

## Principe directeur
On a réfuté 4 « edges » apparents (gate ML 1.88 ; détecteur+filtre 45j ; 4h 150j PF 2.04 ; filtres de features) — tous = artefacts de **direction/régime/queue**. Le risque maintenant n'est plus de manquer un edge, c'est de **se sur-tester nous-mêmes** en ré-analysant en boucle les mêmes données. Ce DOE impose : hypothèses pré-enregistrées, holdout verrouillé touché UNE fois, contrôle des comparaisons multiples, intervalles de confiance bootstrap, stratification par régime, robustesse aux queues.

## Actifs (toutes les données générées)
- `dataset_21d.csv` (224 trades, BEAR, multi-TF) — filtre régime crippled
- `dataset_45d.csv` (578, ~équilibré, multi-TF) — filtre actif
- `dataset_90d_compromised.csv` (259) — filtre OFF (timeline 5j), ne sert que de contre-exemple
- `dataset.csv` courant = **4h / 150j / 947 trades / filtre actif** — le plus profond et propre
- Outils : `build_ml_dataset.py` (backfill fidèle + enrichissement), `evaluate.py` (walk-forward purgé), reconstruction régime (`build_regime_timeline`/`_regime_at`)
- Limite dure : seul le **4h** est backtestable en profondeur (~250j) ; 1h/15m plafonnés (~45-60j) — cf bug fetch profond.

## Règles statistiques non négociables
1. **Métrique = PF/espérance APRÈS COÛTS (0.2% et 0.4% A/R), stratifiée par régime (BULL/BEAR/RANGE), avec IC bootstrap 95%.** Jamais un point estimate seul.
2. **Robustesse aux queues** : tout résultat est recalculé en retirant les jours de krach (jackknife). Un edge qui ne survit pas au jackknife n'est pas un edge.
3. **Holdout cross-sectionnel verrouillé** : symboles séparés en DISCOVERY (13) / HOLDOUT (7, tirés au sort, gelés). Hypothèses formées/tunées sur DISCOVERY uniquement. HOLDOUT touché UNE SEULE FOIS, à la fin.
4. **Walk-forward purgé** (embargo) dans DISCOVERY pour la robustesse temporelle.
5. **Correction multi-tests** : si on teste N facteurs, la barre monte (Bonferroni/FDR). On COMPTE et on LOG chaque hypothèse testée.
6. **Benchmarks obligatoires** : tout edge se mesure contre (a) entrées ALÉATOIRES appariées, (b) take-all, (c) bêta directionnel pur (suivre le régime).

## Phase 0 — Consolidation & splits
- Fusionner tous les datasets en un master tagué (symbole, TF, régime reconstruit, side, run, jours-krach).
- Tirer et GELER le split symboles DISCOVERY/HOLDOUT.
- Tagger les jours de krach (ex : 03-05 fév) pour le jackknife.

## Phase 1 — TEST FONDATEUR (GATE) : le détecteur bat-il l'aléatoire ?
**H1 :** les entrées du détecteur ont une espérance OOS > entrées aléatoires APPARIÉES (même symbole, même side, même stop%/cible%, même régime ; seule la BOUGIE d'entrée diffère).
- Réponse par régime, après coûts, IC bootstrap sur la différence détecteur−aléatoire.
- **GATE :** si l'IC de la différence chevauche 0 dans tous les régimes → les patterns n'apportent RIEN → on arrête la branche « patterns » et on pivote (Phase 3bis). Si > 0 dans ≥1 régime de façon stable → on continue.

## Phase 2 — Décomposer BÊTA vs ALPHA (GATE)
**H2 :** après avoir retiré le bêta directionnel (rendement de « suivre le régime ») ET les jours de krach, il reste une espérance positive.
- Baseline bêta = entrées aléatoires appariées au régime (de Phase 1).
- Alpha = détecteur − bêta, jackknife krach, IC bootstrap.
- **GATE :** alpha résiduel IC>0 ? Sinon, le système = bêta directionnel (pas d'alpha) → décision stratégique (Phase 5).

## Phase 3 — DOE FACTORIEL des features structurelles (si Phase 2 passe)
Facteurs ON/OFF (machinerie déjà présente) : `S/R-break confirmé (clôture HTF + volume)`, `BOS/CHoCH aligné`, `vol_ratio`, `force relative vs BTC`, `distance S/R HTF (ATR)`.
- Plan factoriel 2^k sur DISCOVERY → régression/ANOVA pour isoler les facteurs (et interactions) qui bougent significativement l'espérance.
- Correction multi-tests. Chaque facteur survivant → validé UNE fois sur HOLDOUT.

## Phase 3bis (si Phase 1/2 échouent) — Pivot
Tester un paradigme à support empirique réel : **momentum cross-sectionnel / force relative vs BTC** (classer les symboles, longer les forts / shorter les faibles), indépendant des patterns chartistes.

## Phase 4 — DOE des SORTIES/RISQUE
Facteurs : multiple ATR du stop, RR cible, TP partiel, trailing, time-stop, sizing conditionnel au régime/vol. Factoriel sur DISCOVERY → validation HOLDOUT.

## Phase 5 — VERDICT FINAL verrouillé
Meilleure config des Phases 3-4 → UNE évaluation sur HOLDOUT.
**Critère de succès pré-enregistré : PF holdout ≥ 1.3 après coût 0.4%, borne basse IC95% > 1.0, positif (ou non-négatif) dans ≥2 régimes, survit au jackknife krach.** Sinon : pas d'edge déployable, conclusion honnête.

## Journal des hypothèses testées (anti-data-dredging)
| # | Hypothèse | Set | Résultat | IC95% (diff det−rnd) |
|---|---|---|---|---|
| H1 | Détecteur bat aléatoire apparié (4h/150j) | 4h/150j, 941 det vs 4347 rnd | **REJETÉE hors-krach** : GLOBAL & BULL & BEAR = hasard, RANGE pire que random. Edge brut = krach fév uniquement | GLOBAL hors-krach [−0.94,+0.36] ; BEAR hors-krach [−1.19,+0.95] ; RANGE [−2.70,−0.52] |

| H2 | Momentum cross-sectionnel bat B&H et random (1d, rebal hebdo) | 25 symboles, 300j daily, 34-40 rebal | **INCONCLUANT (sous-puissant)** : aucun lookback/variante ne bat B&H ni random (IC diff ⊃ 0) ; long-only laminé par le bear alt (B&H −54%) ; L/S ≈ random | L=30 L/S vs B&H [−3.3,+3.6]% ; OOS [−1.4,+6.1]% |

## GATE PHASE 1 = ÉCHEC (patterns sans edge). PHASE 3bis = INCONCLUANT (momentum, mais sous-puissant).
## DATA FIX TROUVÉ : **Binance sert 4 ans de daily** (Bitget plafonne ~200 barres en profond). `scripts/momentum_test.py`/`trend_test.py` tirent depuis Binance.

| H2b | Momentum cross-sectionnel sur 4 ANS (Binance, 23 sym, ~200 rebal, multi-cycles) | propre/puissant | **REJETÉE** : L/S (isole l'alpha) = plat/négatif ; long-only ≈ bêta (ne bat pas B&H, IC ⊃ 0). Pas d'alpha momentum | L=30 L/S vs B&H [−2.4,+0.8]% ; OOS no edge |
| H3 | Trend-following (TSMOM long/cash) bat B&H en risque (4 ans, 22 sym) | propre/puissant | **PARTIEL** : pas d'alpha de RENDEMENT (diff IC ⊃ 0) MAIS réduit le DRAWDOWN consistamment (−68%→−42/53% maxDD, Sharpe 0.23→0.36-0.38) et tient OOS (2 moitiés). = gestion du risque, pas prédiction | diff rdt/j [−0.21,+0.21] (non-signif) ; DD/Sharpe robuste |

| H4 | Setup MAÎTRE « sweep liquidité + retournement CHoCH » (SMC) bat aléatoire apparié | Binance 4h, 15 sym, 500j multi-régime, 571 setups ; grille RR×volume | **REJETÉE** : 6 variantes (RR 1.5/2/3 × vol off/>1.5×), AUCUNE ne bat l'aléatoire (GLOBAL = hasard), plusieurs PIRES hors-krach. WR colle au break-even de chaque RR = direction pile-ou-face. Filtre volume n'aide pas | global toutes ⊃ 0 ; hors-krach anti ou ⊃ 0 |

| H5 | La QUALITÉ de détection / DISCRIMINATION (geom_confidence, confluence_score) sépare les gagnants | dataset 4h/150j, 947 trades | **REJETÉE** : in-sample la qualité semble discriminer (Q4 EDGE) mais c'est un artefact directionnel (haute qualité ≈ shorts en bear). OOS (2e moitié) : TOUT négatif et la HAUTE qualité est PIRE (geom Q4 −1.92%, Q3 −3.27%). À direction fixe : pas de gradient monotone. Durcir le tri n'aide pas | OOS quartiles tous E[R]<0 |

| H6 | Variable INDÉPENDANTE (funding rate) discrimine / améliore le PF | Binance perp, 12 sym, 2 ans, 8748 obs (`scripts`/`data/ml/funding_panel.csv`) | **REJETÉE** : Test 1 (fwd par quintile funding) non-monotone, pas de pattern contrarian = bruit. Test 2 (cross-sect long-bas/short-haut) gross Sharpe 1.16 MAIS IC ⊃ 0 (non signif) et NET après frais 0.1%/j = −0.008%/j (négatif). OI Binance trop court (~30j) pour tester | gross [−0.016,+0.204] ; net négatif |

## CONCLUSION DU PROGRAMME : aucun edge de PRÉDICTION/RENDEMENT (patterns larges, ML, momentum X-sect, sweep+CHoCH+volume, tri qualité/discrimination, Bollinger multi-TF, ET la variable indépendante funding — tous rejetés vs aléatoire/OOS/coûts). Variables indépendantes NON testables avec nos données (order-flow L2, on-chain payant, macro externe) = seule frontière restante, mais hors de portée sans nouvelles sources. SEUL résultat robuste = le **trend-following réduit le drawdown** (crisis-alpha, gestion du risque). Objectif réaliste = exposition trend-filtrée + contrôle du risque, PAS « prédire/battre le marché ». Un edge DISCRÉTIONNAIRE (jugement humain non mécanisable) ne peut être validé qu'en **paper-forward** (logger les trades réels et les mesurer vs aléatoire), pas en backtest.
