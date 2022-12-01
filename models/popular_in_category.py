from copy import copy
import typing as t
from datetime import datetime

from rectools.dataset import Dataset, Interactions, features
from rectools.columns import Columns
import pandas as pd

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

            


            