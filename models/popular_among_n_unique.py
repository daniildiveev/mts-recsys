import typing as t

from scipy.stats import mode
import numpy as np
from rectools.dataset import Dataset
from rectools.columns import Columns

from .popular import PopularModel

class PopularAmongNUniqueUsers(PopularModel):
    def fit(self, data:Dataset) -> 'PopularAmongNUniqueUsers':
        interactions = self.cut_data_by_date(data.interactions.df)
        col, agg_func = self.get_params_for_group_by_and_agg(self.popularity_type)

        user_item_matrix = data.get_user_item_matrix()
        top_items = self.get_top_items_covered_users(user_item_matrix)

        filtered_interactions = interactions[interactions[Columns.Item].isin(top_items)]

        item_scores = filtered_interactions.groupby(Columns.Item)[col].agg(agg_func)
        item_scores = item_scores.sort_values(ascending=False)

        items, scores = item_scores.index.values, item_scores.values.astype(np.float32)
        self.popularity_data = (items, scores)

        self._fitted = True

        return self

    @staticmethod
    def get_top_items_covered_users(matrix, n_users=1000) -> np.ndarray:
    
        assert matrix.format == 'csr'

        item_set = []
        covered_users = np.zeros(matrix.shape[0], dtype=bool)
        while covered_users.sum() < n_users: 
            top_item = mode(matrix[~covered_users].indices)[0][0] 
            item_set.append(top_item)
            covered_users += np.maximum.reduceat(matrix.indices==top_item, 
                                                 matrix.indptr[:-1], 
                                                 dtype=bool) 
        return item_set