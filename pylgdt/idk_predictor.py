from .lgdt_predictor import LGDTPredictor


class IDKPredictor(LGDTPredictor):
    def __init__(
        self, min_sup=1, data_structure="sparse_bitset", fit_method="murtree", log=False
    ):
        super().__init__(
            min_sup=min_sup,
            max_depth=0,
            data_structure=data_structure,
            fit_method=fit_method,
            log=log,
        )
