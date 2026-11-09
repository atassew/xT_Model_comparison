# Can the average xT of the last 10 matches predict next-match non-penalty xG?

Three action-valuation methods, all built from scratch from the raw StatsBomb JSON
(no football-specific Python libraries):

* **Karun** — Karun Singh's expected threat (xT)
* **Sumpter** — Sumpter's action-based xT
* **VAEP**

For every team before every match, each method's value is averaged over the team's
last 10 matches, and that average is used to predict the team's non-penalty xG
(npxG) in the next match.

## Data

StatsBomb open data, season 2015/16:

| League               | Matches | Role                                        |
| -------------------- | ------- | ------------------------------------------- |
| Premier League (ENG) | 380     | trained on and forecast                     |
| La Liga (ESP)        | 380     | trained on and forecast                     |
| Serie A (ITA)        | 380     | trained on and forecast                     |
| Ligue 1 (FRA)        | 377     | trained on and forecast                     |
| Bundesliga (GER)     | 34      | training only (too few matches to forecast) |

All three methods are trained on the same data and tested on the same data.

## Setup

* **Same forecasts for every model:** scoring starts at each team's 14th match
  (1,994 forecasts: 500 per league, 494 in FRA). A team needs 10 earlier matches
  for the average, and matches 11–13 are only used as training data until the
  regression has at least 60 rows.
* **Walk-forward, per league:** each league is walked through date by date. Before
  each date, every regression is fitted on that league's rows played *before* the
  date and then predicts the rows *on* the date. So each forecast uses only matches
  played before it. With 20 teams, matchdays 11–13 give exactly 3 × 20 = 60 rows.

## Models

`xT_avg10` is the team's average over its previous 10 matches of one method's
total (Karun, Sumpter or VAEP; passes and crosses only). `npxG_avg10` is the same
average of the team's own npxG.

| Model                          | Formula                                          |
| ------------------------------ | ------------------------------------------------ |
| Baseline (= Model 1)           | npxG_next = a + b × npxG_avg10                   |
| Karun / Sumpter / VAEP         | npxG_next = a + b × xT_avg10                     |
| Model 2 (one per method)       | npxG_next = a + b × npxG_avg10 + c × xT_avg10    |

With the opponent (step 8). `oppNpxGc_avg10` and `oppXTc_avg10` are the npxG and
xT the opponent **conceded** over its own previous 10 matches, i.e. what its last 10
opponents created against it: how weak its defence is.

| Model                          | Formula                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------- |
| Baseline+opp                   | npxG_next = a + b × npxG_avg10 + d × oppNpxGc_avg10                              |
| M3 (one per method)            | npxG_next = a + b × npxG_avg10 + d × oppNpxGc_avg10 + c × xT_avg10 + e × oppXTc_avg10 |

Baseline+opp is the fair yardstick for M3: both know the opponent, so M3 vs
Baseline+opp still asks only whether xT adds anything. That makes 11 models.

Every predictor gets its own regression, so all forecasts are on the npxG scale.
They are compared with RMSE and MAE on unseen matches.

The baseline is the hurdle: a team tends to create about as much npxG as it has
recently. xT might do better because it uses hundreds of passes per match, while
npxG rests on a few shots and is noisier.

## Tests

**Test 1: does xT add anything beyond past npxG?** Model 2 vs Model 1, once per
method (3 methods × 3 tests = 9 results).

| Test                                         | Answers                                             | Role                  |
| -------------------------------------------- | --------------------------------------------------- | --------------------- |
| Clark–West (one-sided, SE clustered by team) | Does adding xT improve forecasts on unseen matches? | Main test             |
| t-test on c (clustered by team)              | Is xT related to next-match npxG beyond past npxG?  | Supporting, in-sample |
| Team-block bootstrap of ΔRMSE                | How big is the improvement, and how certain?        | Effect size + 95% CI  |

Clark–West is built for nested models (Model 2 contains Model 1).

**Test 2: which method is best?** The single-method models (npxG_next = a + b × xT_avg10)
are not nested in each other, so they are compared pairwise with the Diebold–Mariano
test, errors clustered by team (3 results).

| Comparison       | Test            | Answers                                            |
| ---------------- | --------------- | -------------------------------------------------- |
| Karun vs Sumpter | Diebold–Mariano | Which of the two forecasts next-match npxG better? |
| Karun vs VAEP    | Diebold–Mariano | Which of the two forecasts next-match npxG better? |
| Sumpter vs VAEP  | Diebold–Mariano | Which of the two forecasts next-match npxG better? |

**Test 3: the opponent.** The main question is named in advance: M3 vs Baseline+opp.

| Comparison                        | Answers                                                  | Tests                                                   |
| --------------------------------- | -------------------------------------------------------- | ------------------------------------------------------- |
| Baseline+opp vs Baseline          | Does knowing the opponent's defence help?                | Clark–West, bootstrap ΔRMSE, in-sample t-test           |
| **M3 vs Baseline+opp** (per method) | **Does xT still add something once the opponent is known?** | Clark–West, bootstrap ΔRMSE, in-sample joint F (own + opp xT) |
| M3 vs M2 (per method)             | Does adding the opponent improve the xT model?           | Clark–West, bootstrap ΔRMSE, in-sample joint F          |
| M3 Karun / Sumpter / VAEP         | Which method is best with the opponent?                  | Diebold–Mariano, 3 pairs                                |

## How to read the results

| Result                                          | Meaning                                      |
| ----------------------------------------------- | -------------------------------------------- |
| xT beats the baseline and Model 2 beats Model 1 | xT captures attacking quality beyond shots   |
| xT is about equal to the baseline               | It only repeats what past npxG already shows |
| xT is worse than the baseline                   | Past npxG is the better predictor            |

## Training: leave-one-league-out (4 folds)

Karun, Sumpter and VAEP are trained 4 times, each time without the league they value:

| Fold | Train Karun / Sumpter / VAEP on | Value the actions of |
| ---- | ------------------------------- | -------------------- |
| 1    | ESP + ITA + FRA + GER           | ENG                  |
| 2    | ENG + ITA + FRA + GER           | ESP                  |
| 3    | ENG + ESP + FRA + GER           | ITA                  |
| 4    | ENG + ESP + ITA + GER           | FRA                  |

So no method ever values a match it was trained on. Bundesliga is always in the
training set and is never valued or forecast.

## Valued actions: the same for all methods

Only **passes** (open play and kick-offs, as in SPADL) and **crosses** count in a
team's total, for all three methods (`MOVE_TYPES` in `xtn/config.py`). The
comparison is fair: every method totals the same actions. Shots are left out, so
the target (npxG) is never part of the input. Carries, set pieces, dribbles and
defensive actions are not counted.

Why no carries: Sumpter values a move's *position* (a level, not a gain). 83% of
successful passes are followed by a short carry by the receiver (median 3 m), which
gets almost the same value as the pass, so the receiver's spot would be counted
twice (Sumpter's total went from 5.4 to 10.4 per match with carries). Karun's
gains add up to end − start, so carries do not double-count there, but to keep the
same actions for every method, carries are left out of all three totals.

What each method is *fitted* on follows its source: Karun on passes, crosses and
carries (Singh's moves; carries are needed for realistic ball transitions),
Sumpter on passes and crosses (the course's Wyscout "Pass" events), VAEP on all
actions (it needs them as context and for its goal labels).

Each method keeps its own treatment of failed moves, because that is part of its definition:

| Method  | Successful move                  | Failed move                     |
| ------- | -------------------------------- | ------------------------------- |
| Karun   | xT(end zone) − xT(start zone)    | 0 (as in Karun's original)      |
| Sumpter | P(chain ends in shot) × E[xG]    | valued the same way             |
| VAEP    | ΔP(score) − ΔP(concede)          | usually negative (ball lost)    |

## Methods (checked in step 3; scripts in `checks/`)

### Karun xT (`xtn/methods/karun_xt.py`, Singh 2018)

16 × 12 grid; per zone the shot, move and goal probabilities and the move
transitions; solved by value iteration. A successful move gets
xT(end zone) − xT(start zone), a failed move 0. Shots are open-play shots.
Checks (`checks/check_karun.py`): converges in 38 iterations, peak xT ≈ 0.30 in
front of goal, values rise towards goal and are left-right symmetric, the 4 fold
grids differ by at most 0.029 in any zone; same-match correlation of a team's
pass + cross total with npxG 0.55–0.59.

### Sumpter action-based xT (`xtn/methods/sumpter_chain_xt.py`, Soccermatics lesson 4)

Chains = StatsBomb possessions, split after every shot. A chain is labelled
with its shot's xG if it ends with an open-play shot (as in the course, where
penalties and direct free kicks are not "Shot" events). Foul rule: a chain stopped
by an **opponent's** foul gets the label of the same team's directly following
chain (the course applies it to any foul; that would credit a team for the
opponent's shot). Passes and crosses of the team in possession only, 34 polynomial location
features; model A = XGBoost P(shot), model B = linear regression E[xG]; value =
A × B (a level, not a difference, so totals grow with the number of moves).
Differences from the course: StatsBomb possessions and xG, the foul rule above. Checks (`checks/check_sumpter.py`, `tests/test_foul_rule.py`): model A
AUC 0.63–0.64 on the unseen league, model B median 0.09 xG, values rise towards
goal; same-match correlation with npxG 0.34–0.43 (≈ 5.4 per team-match).

### VAEP (`xtn/methods/vaep.py`, Decroos et al. 2019 / socceraction)

Game state = the action and the 2 before it, from the acting team's side;
socceraction's features; labels = the team scores / concedes within 10 actions;
two XGBoost classifiers (100 trees, depth 3, early stopping on AUC);
value = ΔP(score) − ΔP(concede) with socceraction's resets and fixed values before
penalties and corners. Fitted on all actions, but only passes and crosses are summed.
Difference from socceraction: the 25% early-stopping set is whole games instead of
random actions (no game on both sides of the split).
Checks (`checks/check_vaep.py`): AUC on the unseen league 0.81–0.82 (scores) and
0.75–0.81 (concedes); goals ≈ +0.81, own goals ≈ −0.96; successful moves +0.003,
failed moves −0.007; same-match correlation of the pass + cross total with npxG 0.58–0.63.

## Data tables

The converted tables of the parent project (`../data`: games, actions, shots) are
reused. They were checked against this folder's code: re-converting one game per
league from the raw JSON with `xtn/` gives identical actions and shots.

| League | Games | Teams | Matches per team | Mean npxG per team-match |
| ------ | ----- | ----- | ---------------- | ------------------------ |
| ENG    | 380   | 20    | 38               | 1.18                     |
| ESP    | 380   | 20    | 38               | 1.19                     |
| ITA    | 380   | 20    | 38               | 1.09                     |
| FRA    | 377   | 20    | 37–38            | 1.07                     |
| GER    | 34    | 18    | 2–34             | 1.21                     |

3,029,202 actions and 38,728 shots in total. The npxG target is the sum of
StatsBomb xG over a team's shots, without penalties (819 penalties, about 0.78 xG
each) and without shoot-outs (none in these leagues). A team without a shot gets 0
(this happens once).

## Panel (`results/panel.csv`, step 4)

One row per team per match of ENG, ESP, ITA, FRA (3,034 rows): the target `npxg`,
the three method totals over passes and crosses (`Karun`, `Sumpter`, `VAEP`), and
for each of these four columns the average over the team's previous 10 matches
(`*_avg10`). 2,234 rows have an average (every team from its 11th match).
Step 8 adds, for the same four columns, what the team conceded in each match
(`*_conc` = the opponent's value in that game), its 10-match average
(`*_conc_avg10`), and the opponent's conceded average entering the match
(`opp_*_conc_avg10`, the opponent's defence).
Checks (`checks/check_panel.py`): every average was recomputed from the team's 10
earlier matches and is identical (the current match is never included); every
opponent average was recomputed from what the opponent's 10 earlier opponents
created against it and is identical; the VAEP totals are identical to an
independent run.

Correlations between the four 10-match averages:

|               | Karun | Sumpter | VAEP | npxG |
| ------------- | ----- | ------- | ---- | ---- |
| Karun_avg10   | 1.00  | 0.67    | 0.88 | 0.74 |
| Sumpter_avg10 | 0.67  | 1.00    | 0.70 | 0.62 |
| VAEP_avg10    | 0.88  | 0.70    | 1.00 | 0.78 |
| npxg_avg10    | 0.74  | 0.62    | 0.78 | 1.00 |

## Forecast accuracy (`results/metrics.csv`, step 5)

1,994 forecasts (ENG 500, ESP 500, ITA 500, FRA 494), the same rows for every
model, each team from its 14th match; every regression was fitted on at least 60
earlier rows of the same league, with one accepted exception: on FRA's first scored
date (20 Nov 2015) the four opponent models used 58 rows, because FRA's 3 missing
games left two opponents with only 9 earlier matches (2 training rows without an
opponent average). Pooled over the four leagues:

| Model      | RMSE   | MAE    | RMSE − baseline |
| ---------- | ------ | ------ | --------------- |
| Baseline   | 0.6697 | 0.5105 | —               |
| Karun      | 0.6725 | 0.5126 | +0.0028         |
| Sumpter    | 0.6686 | 0.5103 | −0.0010         |
| VAEP       | 0.6699 | 0.5084 | +0.0002         |
| M2 Karun   | 0.6690 | 0.5098 | −0.0007         |
| M2 Sumpter | 0.6647 | 0.5075 | −0.0050         |
| M2 VAEP    | 0.6684 | 0.5076 | −0.0012         |
| Baseline+opp | 0.6556 | 0.5005 | −0.0140       |
| M3 Karun   | 0.6552 | 0.5001 | −0.0145         |
| M3 Sumpter | 0.6505 | 0.4961 | −0.0192         |
| M3 VAEP    | 0.6547 | 0.4973 | −0.0150         |

Without the opponent the differences are small (under 1% of the RMSE). Knowing the
opponent's defence cuts the error by about 2% (Baseline+opp), and M3 Sumpter is the
most accurate model (−2.9%). Whether any of this is more than chance is tested in
step 6 and step 8. Per-league values: `results/metrics.csv`; every
forecast: `results/predictions.csv`; fitted coefficients per date:
`results/coefficients.csv`.

## Test results (step 6)

All tests use the same 1,994 forecasts; standard errors are clustered by team
(80 teams) and the bootstrap resamples whole teams (2,000 times).
Code: `xtn/inference.py`, checked by `tests/test_inference.py`.

**Test 1: Model 2 (past npxG + method) vs Model 1 (past npxG)**
(`results/test1_model2_vs_model1.csv`)

| Method  | Clark–West t | p (one-sided) | c (t-test, in-sample) | p (two-sided) | ΔRMSE   | 95% CI             |
| ------- | ------------ | ------------- | --------------------- | ------------- | ------- | ------------------ |
| Karun   | 3.42         | 0.0003        | 0.451 (t = 3.29)      | 0.0015        | −0.0007 | −0.0052 to +0.0039 |
| Sumpter | 3.92         | < 0.0001      | 0.094 (t = 4.79)      | < 0.0001      | −0.0050 | −0.0101 to +0.0000 |
| VAEP    | 3.79         | 0.0001        | 0.531 (t = 4.69)      | < 0.0001      | −0.0012 | −0.0048 to +0.0023 |

ΔRMSE = RMSE(Model 2) − RMSE(Model 1); negative = Model 2 more accurate. The
t-test fits npxG_next = a_league + b × npxG_avg10 + c × xT_avg10 once on the
scored rows (one intercept per league, as in the walk-forward).

**Test 2: Diebold–Mariano, each pair of methods** (`results/test2_diebold_mariano.csv`;
mean d = MSE(first) − MSE(second), negative = the first is more accurate)

| Models  | Comparison       | mean d  | t     | p (two-sided) |
| ------- | ---------------- | ------- | ----- | ------------- |
| alone   | Karun vs Sumpter | +0.0052 | 0.81  | 0.42          |
| alone   | Karun vs VAEP    | +0.0035 | 1.26  | 0.21          |
| alone   | Sumpter vs VAEP  | −0.0016 | −0.30 | 0.76          |
| Model 2 | Karun vs Sumpter | +0.0058 | 1.51  | 0.13          |
| Model 2 | Karun vs VAEP    | +0.0008 | 0.33  | 0.74          |
| Model 2 | Sumpter vs VAEP  | −0.0050 | −1.59 | 0.11          |

**Test 3: the opponent** (`results/test3_opponent.csv`; ΔRMSE = RMSE(bigger) −
RMSE(smaller), negative = the bigger model is more accurate; F = in-sample joint
test that the added terms are all 0, clustered by team)

| Comparison                 | Added terms               | Clark–West t | p (one-sided) | ΔRMSE   | 95% CI             | F (p)            |
| -------------------------- | ------------------------- | ------------ | ------------- | ------- | ------------------ | ---------------- |
| Baseline+opp vs Baseline   | opp npxG conceded         | 7.41         | < 0.0001      | −0.0140 | −0.0210 to −0.0071 | 80.0 (< 0.0001)  |
| M3 Karun vs Baseline+opp   | own xT + opp xT conceded  | 3.98         | < 0.0001      | −0.0004 | −0.0052 to +0.0048 | 21.1 (< 0.0001)  |
| M3 Sumpter vs Baseline+opp | own xT + opp xT conceded  | 4.72         | < 0.0001      | −0.0052 | −0.0103 to −0.0003 | 18.6 (< 0.0001)  |
| M3 VAEP vs Baseline+opp    | own xT + opp xT conceded  | 3.90         | < 0.0001      | −0.0010 | −0.0052 to +0.0032 | 16.5 (< 0.0001)  |
| M3 Karun vs M2 Karun       | opp npxG + opp xT conceded | 7.69        | < 0.0001      | −0.0138 | −0.0209 to −0.0069 | 48.4 (< 0.0001)  |
| M3 Sumpter vs M2 Sumpter   | opp npxG + opp xT conceded | 8.27        | < 0.0001      | −0.0142 | −0.0210 to −0.0075 | 47.4 (< 0.0001)  |
| M3 VAEP vs M2 VAEP         | opp npxG + opp xT conceded | 7.74        | < 0.0001      | −0.0137 | −0.0209 to −0.0068 | 44.9 (< 0.0001)  |

In M3 (in-sample), both xT terms are significant for every method: own xT
p ≤ 0.002, opponent xT conceded p < 0.0001 (Karun), 0.002 (Sumpter), 0.006 (VAEP).

| Comparison             | mean d  | t     | p (two-sided) |
| ---------------------- | ------- | ----- | ------------- |
| M3 Karun vs M3 Sumpter | +0.0062 | 1.64  | 0.10          |
| M3 Karun vs M3 VAEP    | +0.0007 | 0.28  | 0.78          |
| M3 Sumpter vs M3 VAEP  | −0.0055 | −1.73 | 0.08          |

**Test 4: each method alone vs the baseline** (`results/test4_alone_vs_baseline.csv`;
not nested, so Diebold–Mariano, plus the team-block bootstrap of ΔRMSE)

| Method  | ΔRMSE   | 95% CI             | DM p (two-sided) |
| ------- | ------- | ------------------ | ---------------- |
| Karun   | +0.0028 | −0.0034 to +0.0091 | 0.37             |
| Sumpter | −0.0010 | −0.0088 to +0.0074 | 0.81             |
| VAEP    | +0.0002 | −0.0042 to +0.0046 | 0.92             |

On its own, no method differs clearly from past npxG. `metrics.csv` also has
`R2` = 1 − MSE / var(npxG), the share of match-to-match variation explained
(Baseline 0.064, Baseline+opp 0.103, M3 Sumpter 0.117).

## Build steps

1. Folder and plan ✔
2. Data: raw JSON → games, actions, shots; check match counts and the npxG target ✔
3. Methods: Karun xT, Sumpter action-based xT, VAEP, each checked on its own ✔
4. Panel: per team-match values and the 10-match averages ✔
5. Walk-forward forecasts: baseline, three single-method models, three Model 2s; RMSE / MAE ✔
6. Tests 1 and 2 ✔
7. Results summary ✔ (plain-language report: https://claude.ai/artifact/GNxuAhQS9qSiKnGrFgoo24)
8. Opponent extension ✔
   - Opponent measure: the opponent's xT **conceded**, i.e. the average xT its
     opponents created against it over its previous 10 matches (how weak its
     defence is). Chosen over the opponent's own attacking xT, which relates only
     indirectly to our team's chances.
   - Option 3: keep the 7 models and add Baseline+opp (fair yardstick) and one M3
     per method; the 7 existing results are unchanged.
   - FRA exception: the opponent models used 58 training rows on one date (see
     Forecast accuracy).
   - Added to the plain-language report (step 7): chart, method, results, conclusions.

## Run

```bash
python run.py                    # uses results/team_match_values.csv if present
python run.py --refit            # refit the three methods (about 15 min)
python tests/test_foul_rule.py   # Sumpter chain-label tests
python tests/test_inference.py   # checks of the test statistics
python checks/check_karun.py     # method checks (also check_sumpter, check_vaep, check_panel)
```
