import typing as t
from datetime import datetime

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
        self.ratio = RatioStrategy(ratio)
        
        
