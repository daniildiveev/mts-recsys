import typing as t
from datetime import datetime

import pandas as pd
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

    def cut_data_by_date(self, interactions:pd.DataFrame) -> pd.DataFrame:
        if self.start_from is not None:
            interactions = interactions[interactions[Columns.Datetime] >= self.start_from]
        return interactions

    def fit(self, data:Dataset) -> 'PopularModel':
        pass

    def recommend():
        pass