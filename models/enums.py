from enum import Enum

class Popularity(Enum):
    N_USERS = "n_users"
    N_INTERACTIONS = "n_interactions"
    MEAN_SCORE = "mean_score"
    SUM_SCORE = "sum_score"