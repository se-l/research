import pandas as pd
from abc import ABC

from common.utils.util_func import to_list, convert_enum2str
from common.refdata.curves_reference import CurvesReference as Cr


class PandasFramePlus(pd.DataFrame, ABC):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, feature_hub=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.locp = Loc(self)
        self.feature_hub = feature_hub

    def __getitem__(self, key):
        """could be str -> series or list -> DF"""
        key = convert_enum2str(key)
        to_generate = None
        if to_generate := [c for c in to_list(key) if c not in self.columns]:
            # cols = [resolve_col_name(c)[-1] for c in to_generate]
            to_generate = to_generate[0] if len(to_generate) == 1 else to_generate
            self[to_generate] = self.feature_hub.get_curves(to_generate)
        # keys = [resolve_col_name(c)[-1] for c in to_list(key)]
        # key = keys[0] if len(keys) == 1 else keys
        return pd.DataFrame.__getitem__(self, key)

    def __setitem__(self, key, value):
        pd.DataFrame.__setitem__(self, convert_enum2str(key), value)


class Loc:
    def __init__(self, obj):
        self.obj: pd.DataFrame = obj

    def __getitem__(self, key):
        c = key[-1] if type(key) is tuple else key
        if type(c) is str and c not in self.obj.columns:
            self.obj[c] = 1
        return self.obj.loc[key]


if __name__ == '__main__':
    pdp = PandasFramePlus([1, 2, 3])
    pdp[Cr.close]
    print(pdp.loc[0])
    print(pdp.loc[0, 0])
    print(pdp.locp[:, 'a'])
    print(pdp.locp[0, 'a'])
    print(pdp.locp[0, 'b'])
