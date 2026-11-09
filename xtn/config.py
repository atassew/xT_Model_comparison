"""Central configuration of the study.

Research question
-----------------
Can a team's average xT over its last 10 matches predict its non-penalty xG
(npxG) in the next match, and does it add anything beyond the team's own
past npxG?

Three action-valuation methods are compared: Karun Singh's xT, Sumpter's
action-based xT and VAEP.
"""

# --------------------------------------------------------------------------
# Data: StatsBomb open data, season 2015/16 (season_id = 27)
# competition code -> (competition_id, season_id)
# --------------------------------------------------------------------------
LEAGUES = {
    "ENG": (2, 27),   # Premier League - 380 games (full season)
    "ESP": (11, 27),  # La Liga        - 380 games (full season)
    "ITA": (12, 27),  # Serie A        - 380 games (full season)
    "FRA": (7, 27),   # Ligue 1        - 377 games (near-full season)
    "GER": (9, 27),   # Bundesliga     - only 34 games in the open data
}

# Leagues whose teams are forecast. Each is also one leave-one-league-out fold:
# the methods are trained on all other leagues (GER included) and value this one.
TEST_LEAGUES = ["ENG", "ESP", "ITA", "FRA"]
# Leagues only ever used to train the valuation methods (never valued or forecast).
TRAIN_ONLY_LEAGUES = ["GER"]

# --------------------------------------------------------------------------
# Forecast set-up
# --------------------------------------------------------------------------
WINDOW = 10      # number of previous matches averaged
MIN_TRAIN = 60   # rows the walk-forward regression needs before it forecasts
METHODS = ["Karun", "Sumpter", "VAEP"]

# --------------------------------------------------------------------------
# Pitch and action definitions
# --------------------------------------------------------------------------
FIELD_LENGTH = 105.0  # metres; every action is oriented so the acting team attacks towards x = 105
FIELD_WIDTH = 68.0    # metres

# The only actions that count in a team's total, for all three methods: passes
# (open play and kick-offs, as in SPADL) and crosses. Carries, set pieces,
# dribbles, shots and defensive actions are not counted.
# Carries are left out because Sumpter values a move's position (a level), so a
# short carry after a pass would count the pass's end point a second time.
# What each method is FITTED on follows its source: Karun on passes, crosses and
# carries (Singh's moves), Sumpter on passes and crosses (the course's "Pass"
# events), VAEP on all actions (game-state context and goal labels).
MOVE_TYPES = ["pass", "cross"]

RANDOM_STATE = 123  # seed for every random step (model training, bootstrap)
