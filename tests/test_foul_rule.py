"""Checks of Sumpter's chain labels (``label_chains`` in xtn/methods/sumpter_chain_xt.py).

* A shot always ends a chain (as in the course), and only open-play shots count.
* Foul rule: a chain stopped by a foul gets the shot label of the team's next
  chain, but only if the foul was committed by the OPPONENT and the next chain
  directly follows it.

Run:  python tests/test_foul_rule.py      (or: pytest tests)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xtn.methods.sumpter_chain_xt import label_chains  # noqa: E402

ARS, CHE = 1, 2
NO_XG = np.nan


def game(chains):
    """One test game. chains: list of (possession team, [(acting team, action type, xg), ...])."""
    rows = []
    for pos, (pteam, acts) in enumerate(chains, start=1):
        for team, t, xg in acts:
            rows.append(dict(game_id=1, possession=pos, possession_team_id=pteam,
                             team_id=team, type_name=t, xg=xg))
    return pd.DataFrame(rows)


def labels(df):
    """(shot_end, chain_xg) of every chain, in order."""
    lab = label_chains(df).groupby("chain")[["shot_end", "chain_xg"]].first()
    return [(int(s), round(float(x), 2)) for s, x in lab.itertuples(index=False)]


def action_labels(df):
    """(shot_end, chain_xg) of every action, in order."""
    lab = label_chains(df)
    return [(int(s), round(float(x), 2)) for s, x in zip(lab.shot_end, lab.chain_xg)]


def test_shot_ends_the_chain():
    # one StatsBomb possession: Arsenal shoot (blocked), win the ball back, pass on and lose it.
    # As in the course, the shot ends the chain: the moves before it are labelled,
    # the moves after it start a new chain without a shot.
    df = game([(ARS, [(ARS, "pass", NO_XG), (ARS, "carry", NO_XG), (ARS, "shot", 0.08),
                      (ARS, "carry", NO_XG), (ARS, "pass", NO_XG)])])
    assert action_labels(df) == [(1, 0.08), (1, 0.08), (1, 0.08), (0, 0.0), (0, 0.0)]


def test_two_shots_in_one_possession_are_two_chains():
    df = game([(ARS, [(ARS, "pass", NO_XG), (ARS, "shot", 0.05),
                      (ARS, "cross", NO_XG), (ARS, "shot", 0.40)])])
    assert action_labels(df) == [(1, 0.05), (1, 0.05), (1, 0.40), (1, 0.40)]


def test_opponent_foul_then_shot_is_labelled():
    # Chelsea fouls Arsenal; Arsenal's free kick leads to a shot
    df = game([(ARS, [(ARS, "pass", NO_XG), (ARS, "carry", NO_XG), (CHE, "foul", NO_XG)]),
               (ARS, [(ARS, "freekick_short", NO_XG), (ARS, "pass", NO_XG), (ARS, "shot", 0.12)])])
    assert labels(df) == [(1, 0.12), (1, 0.12)]


def test_own_foul_then_opponent_shot_is_not_labelled():
    # Arsenal fouls; Chelsea's free kick leads to a shot
    df = game([(ARS, [(ARS, "pass", NO_XG), (ARS, "foul", NO_XG)]),
               (CHE, [(CHE, "freekick_short", NO_XG), (CHE, "shot", 0.12)])])
    assert labels(df) == [(0, 0.0), (1, 0.12)]


def test_own_foul_then_own_shot_is_not_labelled():
    # Arsenal fouls but still has the next chain, which ends in a shot
    df = game([(ARS, [(ARS, "pass", NO_XG), (ARS, "foul", NO_XG)]),
               (ARS, [(ARS, "pass", NO_XG), (ARS, "shot", 0.12)])])
    assert labels(df) == [(0, 0.0), (1, 0.12)]


def test_two_opponent_fouls_in_a_row_are_both_labelled():
    df = game([(ARS, [(ARS, "pass", NO_XG), (CHE, "foul", NO_XG)]),
               (ARS, [(ARS, "pass", NO_XG), (CHE, "foul", NO_XG)]),
               (ARS, [(ARS, "pass", NO_XG), (ARS, "shot", 0.30)])])
    assert labels(df) == [(1, 0.30), (1, 0.30), (1, 0.30)]


def test_chain_in_between_blocks_the_label():
    # a Chelsea chain (with no Chelsea action) sits between the foul and the shot
    df = game([(ARS, [(ARS, "pass", NO_XG), (CHE, "foul", NO_XG)]),
               (CHE, [(ARS, "interception", NO_XG)]),
               (ARS, [(ARS, "pass", NO_XG), (ARS, "shot", 0.20)])])
    assert labels(df) == [(0, 0.0), (0, 0.0), (1, 0.20)]


def test_penalty_and_direct_free_kick_are_not_shots():
    # as in the course, only open-play shots count: a penalty or a direct free-kick
    # shot does not give its chain (or the fouled chain before it) a shot label
    df = game([(ARS, [(ARS, "pass", NO_XG), (CHE, "foul", NO_XG)]),
               (ARS, [(ARS, "shot_penalty", 0.78)]),
               (ARS, [(ARS, "pass", NO_XG), (CHE, "foul", NO_XG)]),
               (ARS, [(ARS, "shot_freekick", 0.06)])])
    assert labels(df) == [(0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)]


def test_opponent_foul_without_later_shot_is_not_labelled():
    df = game([(ARS, [(ARS, "pass", NO_XG), (CHE, "foul", NO_XG)]),
               (ARS, [(ARS, "pass", NO_XG), (ARS, "pass", NO_XG)])])
    assert labels(df) == [(0, 0.0), (0, 0.0)]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
