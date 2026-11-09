"""Convert raw StatsBomb events into a table of on-the-ball actions.

This replaces socceraction's SPADL conversion. The action types are SPADL's
(socceraction/spadl/config.py and its StatsBomb converter), derived here
directly from the StatsBomb event JSON, with one addition: an own goal is its
own type ``own_goal`` (SPADL stores it as a bad_touch with result owngoal,
which its VAEP labels then miss). SPADL's ``non_action`` (deleted after
conversion) and ``keeper_pick_up`` (never produced from StatsBomb data) are
not needed, so the vocabulary has 22 types.

Coordinates
-----------
StatsBomb uses a 120 x 80 pitch and records every event from the point of
view of the team performing it, attacking from left (x = 0) to right
(x = 120). Coordinates are rescaled to metres on a 105 x 68 pitch, keeping
that orientation, so **every action is already left-to-right for the acting
team** and no mirroring is needed.

Action types and results
------------------------
=====================  =========================================  ======================
StatsBomb event        condition                                  action type
=====================  =========================================  ======================
Pass                   pass.type = Throw-in                       throw_in
                       pass.type = Corner, high pass or cross     corner_crossed
                       pass.type = Corner, otherwise              corner_short
                       pass.type = Free Kick, high pass or cross  freekick_crossed
                       pass.type = Free Kick, otherwise           freekick_short
                       pass.type = Goal Kick                      goalkick
                       pass.cross = true (other passes)           cross
                       otherwise (open play and kick-offs)        pass
Carry                                                             carry
Dribble                                                           take_on
Shot                   shot.type = Penalty / Free Kick / other    shot_penalty / shot_freekick / shot
Foul Committed                                                    foul
Duel                   duel.type = Tackle                         tackle
Interception                                                      interception
Clearance                                                         clearance
Miscontrol                                                        bad_touch
Goal Keeper            Shot Saved*, Save, Smother /               keeper_save /
                       Collected, Keeper Sweeper / Punch          keeper_claim / keeper_punch
Own Goal Against                                                  own_goal
=====================  =========================================  ======================

Result = "success" when: a pass has no outcome; a carry (always); a take-on
is Complete; a shot is a Goal; a tackle/interception is Won or Success*; a
clearance (always); a keeper action unless its outcome is Fail / No Touch.
Pass offside -> "offside", own goal -> "owngoal", everything else -> "fail".
Fouls and bad touches are always "fail".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SB_LENGTH, SB_WIDTH = 120.0, 80.0
FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0

ACTION_TYPES = [
    "pass", "cross", "throw_in", "freekick_crossed", "freekick_short", "corner_crossed",
    "corner_short", "goalkick", "carry", "take_on", "shot", "shot_freekick", "shot_penalty",
    "foul", "tackle", "interception", "clearance", "bad_touch", "keeper_save",
    "keeper_claim", "keeper_punch", "own_goal",
]
RESULTS = ["fail", "success", "offside", "owngoal"]
BODYPARTS = ["foot", "head", "other"]

_SETPIECE_PASS = {"Throw-in": "throw_in", "Goal Kick": "goalkick"}
_CROSSED_OR_SHORT = {"Corner": "corner", "Free Kick": "freekick"}  # -> <name>_crossed / _short
_KEEPER = {"Shot Saved": "keeper_save", "Shot Saved Off Target": "keeper_save",
           "Shot Saved to Post": "keeper_save", "Save": "keeper_save", "Smother": "keeper_save",
           "Collected": "keeper_claim", "Keeper Sweeper": "keeper_claim", "Punch": "keeper_punch"}
_WON = {"Won", "Success", "Success In Play", "Success Out"}


def _name(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _bodypart(name):
    if not name:
        return "foot"
    if "Head" in name:
        return "head"
    if "Foot" in name:
        return "foot"
    return "other"


def _seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _classify(e: dict):
    """Return (type, result, bodypart, end_location) or None to skip the event."""
    t = e["type"]["name"]
    loc = e.get("location")
    if t == "Pass":
        p = e["pass"]
        sp = _name(p, "type", "name")
        if sp in _SETPIECE_PASS:
            atype = _SETPIECE_PASS[sp]
        elif sp in _CROSSED_OR_SHORT:
            # SPADL's rule: a high pass or a cross is "crossed", anything else "short"
            crossed = _name(p, "height", "name") == "High Pass" or p.get("cross")
            atype = _CROSSED_OR_SHORT[sp] + ("_crossed" if crossed else "_short")
        else:  # open play and kick-offs (SPADL treats a kick-off as a pass)
            atype = "cross" if p.get("cross") else "pass"
        out = _name(p, "outcome", "name")
        res = "success" if out is None else ("offside" if out == "Pass Offside" else "fail")
        return atype, res, _bodypart(_name(p, "body_part", "name")), p.get("end_location")
    if t == "Carry":
        return "carry", "success", "foot", e["carry"].get("end_location")
    if t == "Dribble":
        res = "success" if _name(e, "dribble", "outcome", "name") == "Complete" else "fail"
        return "take_on", res, "foot", loc
    if t == "Shot":
        s = e["shot"]
        st = _name(s, "type", "name")
        atype = {"Penalty": "shot_penalty", "Free Kick": "shot_freekick"}.get(st, "shot")
        res = "success" if _name(s, "outcome", "name") == "Goal" else "fail"
        return atype, res, _bodypart(_name(s, "body_part", "name")), s.get("end_location")
    if t == "Foul Committed":
        return "foul", "fail", "foot", loc
    if t == "Duel":
        if _name(e, "duel", "type", "name") != "Tackle":
            return None
        res = "success" if _name(e, "duel", "outcome", "name") in _WON else "fail"
        return "tackle", res, "foot", loc
    if t == "Interception":
        res = "success" if _name(e, "interception", "outcome", "name") in _WON else "fail"
        return "interception", res, "foot", loc
    if t == "Clearance":
        return "clearance", "success", _bodypart(_name(e, "clearance", "body_part", "name")), loc
    if t == "Miscontrol":
        return "bad_touch", "fail", "foot", loc
    if t == "Goal Keeper":
        gt = _name(e, "goalkeeper", "type", "name")
        if gt not in _KEEPER:
            return None  # Shot Faced, Goal Conceded, Penalty Conceded: not actions
        out = _name(e, "goalkeeper", "outcome", "name")
        res = "fail" if out in ("Fail", "No Touch") else "success"
        return _KEEPER[gt], res, "other", loc
    if t == "Own Goal Against":
        return "own_goal", "owngoal", "other", loc
    return None


def convert(events: list[dict], game_id: int):
    """Convert one match's events. Returns (actions, shots) DataFrames."""
    events = sorted(events, key=lambda e: e["index"])
    rows, shots = [], []
    for e in events:
        if e.get("period", 0) >= 5:  # penalty shoot-out
            continue
        if e["type"]["name"] == "Shot":
            shots.append(dict(
                game_id=game_id, event_id=e["id"], team_id=e["team"]["id"], period_id=e["period"],
                minute=e["minute"], xg=_name(e, "shot", "statsbomb_xg"),
                is_penalty=_name(e, "shot", "type", "name") == "Penalty",
            ))
        c = _classify(e)
        if c is None:
            continue
        atype, res, body, end = c
        start = e.get("location") or end or [SB_LENGTH / 2, SB_WIDTH / 2]
        end = end or start
        rows.append(dict(
            game_id=game_id, event_id=e["id"], event_index=e["index"], period_id=e["period"],
            time_seconds=_seconds(e["timestamp"]), team_id=e["team"]["id"],
            player_id=_name(e, "player", "id"),
            start_x=start[0], start_y=start[1], end_x=end[0], end_y=end[1],
            type_name=atype, result_name=res, bodypart_name=body,
            possession=e["possession"], possession_team_id=e["possession_team"]["id"],
            xg=_name(e, "shot", "statsbomb_xg") if e["type"]["name"] == "Shot" else np.nan,
        ))
    a = pd.DataFrame(rows)
    # metres on a 105 x 68 pitch, clipped to the pitch
    for c, scale, lim in [("start_x", FIELD_LENGTH / SB_LENGTH, FIELD_LENGTH),
                          ("end_x", FIELD_LENGTH / SB_LENGTH, FIELD_LENGTH),
                          ("start_y", FIELD_WIDTH / SB_WIDTH, FIELD_WIDTH),
                          ("end_y", FIELD_WIDTH / SB_WIDTH, FIELD_WIDTH)]:
        a[c] = (a[c].astype(float) * scale).clip(0, lim)
    a["action_id"] = np.arange(len(a))
    return a, pd.DataFrame(shots)
