from copy import copy
from operator import index
import typing as t
from datetime import datetime

from rectools.dataset import Dataset, Interactions, features
from rectools.columns import Columns
import pandas as pd
import numpy as np

from .popular import PopularModel
from .enums import MixingStrategy, RatioStrategy

class PopularInCategory(PopularModel):
    def __init__(self,
                 category:str,
                 n_categories:t.Optional[int]=None,
                 ratio:t.Optional[str]='proportional',
                 mixing:t.Optional[str]='rotate',
                 popularity_type:t.Optional[str]='n_users',
                 start_from:t.Optional[datetime]=None) -> None:
        super().__init__(popularity_type, start_from)

        self.category = category
        self.n_categories = n_categories
        self.mixing = MixingStrategy(mixing)
        self.cat_interactions = {}
        self.ratio = RatioStrategy(ratio)
        self.cat_columns = []
        self.models = {}
        
    def validate_cat_feature(self, data:Dataset) -> None:
        if not data.item_features:
            raise ValueError("specify item features when creating dataset")

        for i, (k, v) in enumerate(data.item_features):
            if k == self.category:
                self.cat_columns.append(i)

        if self.cat_columns == []:
            raise ValueError("feature not in data")
        
    def score_category(self, data:Dataset, interactions:pd.DataFrame):
        scores = {}

        for col_num in self.cat_columns:
            item_id = data.item_features.values.getcol(col_num).nonzero()[0]
            interaction = copy(interactions[interactions[Columns.Item].isin(item_id)])

            if not interaction.shape[0] == 0:
                self.cat_interactions[col_num] = interaction
                
                col, agg_func = self.get_params_for_group_by_and_agg(self.popularity_type)

                scores[col_num] = self.cat_interactions[col_num][col].apply(agg_func)
        
        self.cat_scores = pd.Series(scores).sort_values(ascending=False)

    def define_categories(self) -> None:
        if self.n_categories:
            if len(self.cat_columns) >= self.n_categories:
                self.n_effective_cats = self.n_categories
                relevant_cats = self.cat_scores.iloc[:self.n_categories].index
                self.cat_scores = self.cat_scores.loc[relevant_cats]
                self.cat_columns = relevant_cats
        
            else:
                self.n_effective_cats = len(self.cat_columns)

        else:
            self.n_effective_cats = len(self.cat_columns)

    def fit(self, data:Dataset) -> 'PopularInCategory':
        self.validate_cat_feature(data)

        interactions = self.cut_data_by_date(data.interactions.df)
        self.score_category(data, interactions)
        self.define_categories()

        for col_num in self.cat_columns:
            cat_interactions = Interactions(self.cat_interactions[col_num])
            cat_dataset = Dataset(
                data.user_id_map, data.item_id_map, cat_interactions
            )

            cat_model = PopularModel(
                popularity_type=self.popularity_type
            ).fit(cat_dataset)

            self.models[col_num] = cat_model

    def get_num_recs_for_each_category(self, n:int) -> pd.Series:
        if self.ratio == RatioStrategy.EQUAL:
            ns_of_cols = self.score_category.index
            n_recs = pd.Series({n_col: n // self.n_effective_cats for n_col in ns_of_cols})
            surplus_recs = n - np.sum(n_recs)
            n_recs.iloc[:surplus_recs] += 1

        elif self.ratio == RatioStrategy.PROPORTIONAL:
            scores_sum = np.sum(self.cat_scores)
            n_recs = np.floor(self.cat_scores * n  / scores_sum).astype(np.int32)
            surplus_recs = n - n_recs.sum()
            n_recs.iloc[surplus_recs] += 1

            zero_mask = (n_recs == 0)
            may_be_decreased_mask = (n_recs > 1)
            num_changing_zeros = np.min(np.sum(may_be_decreased_mask), np.sum(zero_mask))

            if num_changing_zeros > 1:
                indexes_to_increase = np.arange(len(n_recs))[zero_mask][:num_changing_zeros]
                indexes_to_decrease = np.arange(len(n_recs))[may_be_decreased_mask][:-num_changing_zeros]

                n_recs.iloc[indexes_to_increase] += 1
                n_recs.iloc[indexes_to_decrease] -= 1

        return n_recs
            