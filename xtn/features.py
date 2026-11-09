"""From action values to the forecasting table ("panel").

Pipeline in this module
-----------------------
1. ``team_match_xg``      the target: non-penalty xG of every team in every match
2. ``team_match_values``  the sum of a method's action values for every team in every match
3. ``build_panel``        joins 1 and 2 and adds, for every team and match, the
                          average of the team's OWN previous 10 matches

The key line is in ``build_panel``:

    groupby("team_id") ... shift(1).rolling(10).mean()

* ``groupby("team_id")``  -> each team's average uses only that team's matches
* sorted by date          -> "previous" means earlier in the season
* ``shift(1)``            -> the match being forecast is never part of its own input
* ``min_periods=10``      -> no value until the team has played 10 matches
"""
from __future__ import annotations

import pandas as pd

from . import config


def team_match_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, one for each team, with its opponent and the date.

    Example: Arsenal v Chelsea becomes
        (game, Arsenal, opponent = Chelsea) and (game, Chelsea, opponent = Arsenal)
    """
    g = games[["game_id", "league", "game_date", "home_team_id", "away_team_id"]]
    home = g.rename(columns={"home_team_id": "team_id", "away_team_id": "opp_id"})
    away = g.rename(columns={"away_team_id": "team_id", "home_team_id": "opp_id"})
    return pd.concat([home, away], ignore_index=True)


def team_match_xg(shots: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Target variable: non-penalty xG (npxG) of each team in each match.

    * StatsBomb's ``shot.statsbomb_xg`` is summed over the team's shots.
    * Penalties are left out (winning one is mostly chance, and each is worth ~0.78).
    * Penalty shoot-outs (period 5) are left out.
    * A team without a shot gets 0.
    """
    s = shots[(~shots.is_penalty) & (shots.period_id < 5)]
    xg = s.groupby(["game_id", "team_id"]).xg.sum().rename("npxg").reset_index()
    tm = team_match_frame(games).merge(xg, on=["game_id", "team_id"], how="left")
    tm["npxg"] = tm.npxg.fillna(0.0)
    return tm


def team_match_values(actions: pd.DataFrame, values: pd.Series, games: pd.DataFrame,
                      name: str, types=config.MOVE_TYPES) -> pd.DataFrame:
    """Sum one method's action values per team per match.

    Parameters
    ----------
    actions : the actions that were rated (one row per action, with game_id and team_id)
    values  : one value per action, same index as ``actions`` (NaN = not valued)
    name    : column name for the result, e.g. "Karun"
    types   : the action types that count (default: passes and crosses, for every
        method). All other actions are left out of the sum, so the three methods
        total the same actions.

    Returns
    -------
    DataFrame with columns game_id, team_id, <name>: the total value of that team's
    OWN passes and crosses in that match. Each action is credited to the team
    that performed it; every team of every game in ``actions`` gets a row.
    """
    v = values.fillna(0.0)  # actions a method does not value count as 0
    v = v.where(actions.type_name.isin(types), 0.0)
    df = pd.DataFrame({"game_id": actions.game_id, "team_id": actions.team_id, name: v})
    agg = df.groupby(["game_id", "team_id"])[name].sum().reset_index()

    # make sure every team in every rated game has a row (0 if it had no valued action)
    tm = team_match_frame(games[games.game_id.isin(actions.game_id.unique())])[["game_id", "team_id"]]
    tm = tm.merge(agg, on=["game_id", "team_id"], how="left")
    tm[name] = tm[name].fillna(0.0)
    return tm


def build_panel(xg_tm: pd.DataFrame, values_tm: pd.DataFrame, window: int = config.WINDOW) -> pd.DataFrame:
    """One row per team per match, with the target and the 10-match averages.

    Parameters
    ----------
    xg_tm     : output of ``team_match_xg`` (forecast leagues only)
    values_tm : game_id, team_id and one column per method (Karun, Sumpter, VAEP)

    Columns added, for c in Karun, Sumpter, VAEP and npxg:

        c_avg10           = mean of c over the team's previous 10 matches (current match excluded)
        c_conc            = c of the OPPONENT in this match, i.e. what the team conceded
        c_conc_avg10      = mean of c_conc over the team's previous 10 matches
        opp_c_conc_avg10  = the opponent's c_conc_avg10 in this match: how much the
                            opponent's last 10 opponents created against it (its defence)

    plus ``match_no`` (1 = the team's first match of the season).

    Example, Arsenal v Chelsea: Arsenal's row gets Chelsea's Karun_conc_avg10 as
    opp_Karun_conc_avg10, i.e. the Karun xT that Chelsea's previous 10 opponents
    created against Chelsea. Chelsea's average excludes this match, so Arsenal's
    own xT in it is never an input.
    """
    panel = xg_tm.merge(values_tm, on=["game_id", "team_id"], how="left", validate="one_to_one")
    assert panel[config.METHODS].notna().all().all(), "a team-match has no method value"
    cols = config.METHODS + ["npxg"]

    # what each team conceded = what its opponent created in the same game
    opp = panel[["game_id", "team_id"] + cols].rename(columns={"team_id": "opp_id", **{c: f"{c}_conc" for c in cols}})
    panel = panel.merge(opp, on=["game_id", "opp_id"], how="left", validate="one_to_one")

    # Each team's matches in date order (game_id breaks ties).
    panel = panel.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    per_team = panel.groupby("team_id")
    panel["match_no"] = per_team.cumcount() + 1
    for c in cols + [f"{c}_conc" for c in cols]:
        # shift(1): drop the current match; rolling(window): average of the 10 before it.
        panel[f"{c}_avg{window}"] = per_team[c].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean())

    # the opponent's defence entering this match
    conc = [f"{c}_conc_avg{window}" for c in cols]
    opp_avg = panel[["game_id", "team_id"] + conc].rename(columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in conc}})
    return panel.merge(opp_avg, on=["game_id", "opp_id"], how="left", validate="one_to_one")
