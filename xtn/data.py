"""Build and load the analysis tables from StatsBomb open data.

Outputs (in ``data_dir``):
    raw/                 cached raw StatsBomb JSON (gzip)
    games.parquet        one row per game (league, date, home/away team ids)
    actions.parquet      on-the-ball actions of all games (see xtn/actions.py), with
                         possession, possession_team_id and xg (shots only)
    shots.parquet        one row per shot with StatsBomb xG and a penalty flag
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from . import actions as act
from . import config, statsbomb


def _games_table(raw_dir: str, leagues) -> pd.DataFrame:
    rows = []
    for lg in leagues:
        cid, sid = config.LEAGUES[lg]
        for m in statsbomb.matches(raw_dir, cid, sid):
            rows.append(dict(
                game_id=m["match_id"], league=lg, competition_id=cid, season_id=sid,
                game_date=pd.Timestamp(f"{m['match_date']} {m.get('kick_off') or '00:00:00'}"),
                home_team_id=m["home_team"]["home_team_id"], away_team_id=m["away_team"]["away_team_id"],
                home_team_name=m["home_team"]["home_team_name"], away_team_name=m["away_team"]["away_team_name"],
                home_score=m["home_score"], away_score=m["away_score"], match_week=m.get("match_week"),
            ))
    return pd.DataFrame(rows)


def _convert_one(args):
    raw_dir, game_id = args
    return act.convert(statsbomb.events(raw_dir, game_id), game_id)


def build(data_dir: str, leagues=None, workers: int = 2):
    """Download (if needed) and convert all games of the given leagues."""
    leagues = leagues or list(config.LEAGUES)
    raw_dir = os.path.join(data_dir, "raw")
    games = _games_table(raw_dir, leagues)
    print(f"{len(games)} games; downloading events not yet cached ...", flush=True)
    statsbomb.download_events(raw_dir, games.game_id.tolist())

    print("converting events to actions ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_convert_one, [(raw_dir, g) for g in games.game_id], chunksize=8))
    actions = pd.concat([r[0] for r in res], ignore_index=True)
    shots = pd.concat([r[1] for r in res], ignore_index=True)

    lg = games.set_index("game_id").league
    actions["league"] = actions.game_id.map(lg)
    shots["league"] = shots.game_id.map(lg)
    actions = actions.sort_values(["game_id", "action_id"]).reset_index(drop=True)

    games.to_parquet(os.path.join(data_dir, "games.parquet"), index=False)
    actions.to_parquet(os.path.join(data_dir, "actions.parquet"), index=False)
    shots.to_parquet(os.path.join(data_dir, "shots.parquet"), index=False)
    print(f"saved {len(games)} games, {len(actions):,} actions, {len(shots):,} shots")


def load(data_dir: str):
    """Load the cached tables with compact dtypes."""
    games = pd.read_parquet(os.path.join(data_dir, "games.parquet"))
    actions = pd.read_parquet(os.path.join(data_dir, "actions.parquet"))
    shots = pd.read_parquet(os.path.join(data_dir, "shots.parquet"))
    games["game_date"] = pd.to_datetime(games.game_date)
    actions = actions.drop(columns=["event_id"])
    actions["type_name"] = pd.Categorical(actions.type_name, categories=act.ACTION_TYPES)
    actions["result_name"] = pd.Categorical(actions.result_name, categories=act.RESULTS)
    actions["bodypart_name"] = pd.Categorical(actions.bodypart_name, categories=act.BODYPARTS)
    actions["league"] = actions.league.astype("category")
    for c in ["start_x", "start_y", "end_x", "end_y", "time_seconds", "xg"]:
        actions[c] = actions[c].astype("float32")
    return games, actions, shots
