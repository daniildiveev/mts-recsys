import typing as t
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from rectools import Columns
from rectools.dataset.dataset import Dataset

from .enums import Popularity

class PopularModel:
    def __init__(self,
                 popularity_type:str,
                 start_from:t.Optional[datetime],
                 ) -> None:

        self.start_from = start_from

        try: 
            self.popularity_type = Popularity(popularity_type)
        except ValueError:
            raise ValueError("there is no such option: %s" % popularity_type)

        self._fitted = False

    def cut_data_by_date(self, interactions:pd.DataFrame) -> pd.DataFrame:
        if self.start_from is not None:
            interactions = interactions[interactions[Columns.Datetime] >= self.start_from]
        return interactions

    @staticmethod
    def get_params_for_group_by_and_agg(popularity_type:Popularity) -> t.Tuple[str, str]:
        if popularity_type == Popularity.N_USERS:
            return Columns.User, 'nunique'
        elif popularity_type == Popularity.N_INTERACTIONS:
            return Columns.User, 'count'
        elif popularity_type == Popularity.MEAN_SCORE:
            return Columns.Weight, 'mean'
        elif popularity_type == Popularity.SUM_SCORE:
            return Columns.Weight, 'sum'

    def fit(self, data:Dataset) -> 'PopularModel':
        interactions = self.cut_data_by_date(data.interactions.df)

        col, agg_func = self.get_params_for_group_by_and_agg(self.popularity_type)
        item_scores = interactions.groupby(Columns.Item)[col].agg(agg_func)
        item_scores = item_scores.sort_values(ascending=False)

        items, scores = item_scores.index.values, item_scores.values.astype(np.float32)
        self.popularity_data = (items, scores)

        self._fitted = True

        return self

    def recommend(self,
                  users_ids: np.ndarray,
                  k:int,
                  ) -> pd.DataFrame: 

        user_ids_all, reco_ids_all, scores_all = [], [], []

        for user_id in tqdm(users_ids, desc='Getting recommendations '):
            reco_ids, scores = self.popularity_data[:, :k]
            
            user_ids_all += [user_id] * k
            reco_ids_all += reco_ids
            scores_all += scores

        return pd.DataFrame(data={
            Columns.User : user_ids_all,
            Columns.Item : reco_ids_all,
            Columns.Weight : scores_all
        })

