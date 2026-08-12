import os
import json
import pickle
import pandas as pd
import numpy as np
import io

from functools import partial
from sklearn.preprocessing import KBinsDiscretizer
from concurrent.futures import ProcessPoolExecutor
from shared.modules.logger import logger
from common.paths import Paths
from common.utils.util_func import get_feature_names, create_dir
from common.utils.util_func import default_to_py_type
from trader.data_loader.utils_features import digitize


class Normalize:
    """
    Either bin or scale inputs. Load and store values used for normalization
    Each dataset gets their respective instance for normalization
    """
    ix_min = 0
    ix_max = 1
    rb = {'KBins': 'b', 'json': ''}

    def __init__(s, overwrite_stored=False, **kwargs):
        s.overwrite_stored = overwrite_stored
        s.ex = kwargs.get('ex')
        s.range_fn = kwargs.get('range_fn')
        s.store = None

    def ready_norm_min_max(s, arr):
        s.store = {col: (None, None) for col in get_feature_names(arr)}
        if s.store_exists('json'):
            s.store.update(s.load_normalize_store('json'))

    def temp_save(s, col, arr_min, arr_max):
        s.store[col] = (default_to_py_type(arr_min), default_to_py_type(arr_max))

    def normalize_scale01_ndarr(s, arr):
        s.ready_norm_min_max(arr)
        for col in get_feature_names(arr):
            arr[col], arr_min, arr_max = s.normalize_scale01_1darr(arr[col], s.store[col][s.ix_min], s.store[col][s.ix_max])
            s.temp_save(col, arr_min, arr_max)
        s.store_normalized('json')
        return arr

    def normalize_kmeans_bin_ndarr(s, arr, n_bins=30):
        s.store = s.load_normalize_store('KBins') if s.store_exists('KBins') else {}
        if len(arr) < 5000:
            for col in get_feature_names(arr):
                if col in s.store.keys() and not s.overwrite_stored:
                    continue
                est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
                if type(arr) == pd.DataFrame:
                    est.fit(arr[col].values.reshape(-1, 1))
                else:
                    est.fit(arr[col].reshape(-1, 1))
                s.store[col] = est
        else:
            print('Calculating KMean for KBins...')
            norm = [c for c in arr.columns if c not in s.store.keys()]
            with ProcessPoolExecutor(max_workers=min(4, len(arr[norm].columns))) as pool:
                est_name = list(pool.map(partial(s.pp_kbin_transform, n_bins=n_bins), s.arr_num_tup_gen(arr[norm])))
            s.store.update({name: est for est, name in est_name})
        print('Applying KBins...')
        for col in get_feature_names(arr):
            if col not in s.store.keys():
                continue
            if type(arr) == pd.DataFrame:
                arr[col] = s.store[col].transform(arr[col].values.reshape(-1, 1)).reshape(1, -1)[0].astype(int)
            else:
                arr[col] = s.store[col].transform(arr[col].reshape(-1, 1)).reshape(1, -1)[0].astype(int)
        s.store_normalized('KBins')
        return arr

    def normalize_kbin_kwargs(s, values: list, feature_names: list):
        """those must have a matching sequence"""
        for i in range(len(feature_names)):
            col = feature_names[i]
            if col not in s.store.keys():
                continue
            else:
                values[i] = s.store[col].transform(np.array([values[i]]).reshape(-1, 1)).reshape(1, -1)[0].astype(int)

    @classmethod
    def normalize_scale01_ndarr_static(cls, arr):
        for col in range(arr.shape[1]):
            if type(arr) == np.ndarray:
                arr[:, col], arr_min, arr_max = cls.normalize_scale01_1darr(arr[:, col])
            elif type(arr) == pd.DataFrame:
                arr.iloc[:, col], arr_min, arr_max = cls.normalize_scale01_1darr(arr.iloc[:, col])
            else:
                raise TypeError('array type not understood')
        return arr

    @staticmethod
    def normalize_scale01_1darr(arr, arr_min=None, arr_max=None):
        if arr_min is None:
            arr_min = np.min(arr)
        if arr_max is None:
            arr_max = np.max(arr)
        return np.divide(
            np.subtract(arr, arr_min),
            np.subtract(arr_max, arr_min)
        ), arr_min, arr_max

    def store_exists(s, type_):
        return os.path.exists(os.path.join(Paths.trade_model, s.ex, f'{s.range_fn}.{type_}'))

    def store_normalized(s, type_):
        if s.store_exists(type_):
            current_dct = s.load_normalize_store(type_)
            current_dct.update(s.store)
            s.store = current_dct
        create_dir(os.path.join(Paths.trade_model, s.ex))
        with open(os.path.join(Paths.trade_model, s.ex, f'{s.range_fn}.{type_}'), f'w{s.rb[type_]}') as out:
            return s.io_f(out, s.store)

    @staticmethod
    def io_f(f, data=None):
        if isinstance(f, io.TextIOWrapper):
            return json.dump(data, f) if data else json.load(f)
        elif isinstance(f, (io.BufferedWriter, io.BufferedReader)):
            return pickle.dump(data, f) if data else pickle.load(f)

    def load_normalize_store(s, type_):
        with open(os.path.join(Paths.trade_model, s.ex, f'{s.range_fn}.{type_}'), f'r{s.rb[type_]}') as f:
            return s.io_f(f)

    @staticmethod
    def float_to_int_approx(arr, digits=6):
        """
        convert 0-1 scaled inputs to an integer representation to save memory and space
        """
        return np.multiply(arr, 10 ** digits).astype(int)

    def quantize_df(s, df_full: pd.DataFrame):
        try:
            s.store = s.load_normalize_store('json')
            for c in df_full.columns:
                df_full[c] = np.digitize(
                    df_full[c],
                    bins=s.store[c],
                    right=True
                )
        except FileNotFoundError:
            print('Creating new bins.txt...')
            df_full, s.store = digitize(df_full, df_full.columns, n_bins=20.1)
            s.store_normalized('json')
        return df_full

    def kmeans_bin_df(s, df_full: pd.DataFrame, n_bins=20, sample_size=100000):
        s.store = s.load_normalize_store('KBins') if s.store_exists('KBins') else {}
        set_store_norm = [c for c in df_full.columns if c not in s.store.keys()]
        if len(df_full) < 5000 and set_store_norm:
            for c in df_full.columns:
                if c in s.store.keys() and not s.overwrite_stored:
                    continue
                est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
                est.fit(df_full[c].values.reshape(-1, 1))
                s.store[c] = est
        elif set_store_norm:
            logger.info(f'Calculating -> Applying KMean for KBins for columns: {set_store_norm} ...')
            if len(set_store_norm) > 1:
                with ProcessPoolExecutor(max_workers=min(4, len(df_full[set_store_norm].columns))) as pool:
                    est_name = list(pool.map(partial(s.pp_kbin_transform, n_bins=n_bins), s.arr_num_tup_gen(df_full[set_store_norm], sample_size)))
            else:
                est_name = [s.pp_kbin_transform(tup, n_bins=20) for tup in s.arr_num_tup_gen(df_full[set_store_norm], sample_size)]
            s.store.update({name: est for est, name in est_name})
        s.store_normalized('KBins')
        # use_store_norm = [c for c in df_full.columns if c in s.store.keys()]
        for c in df_full.columns:
            df_full[c] = s.store[c].transform(df_full[c].values.reshape(-1, 1)).reshape(1, -1)[0].astype(np.int8)
        return df_full

    def kmeans_bin_nda(s, nda: np.array, col_name: str, n_bins=20, sample_size=100000):
        # logger.info(f'Calculating -> Applying KMean for KBins for columns: {col_name} ...')
        fitted = col_name in s.store.keys() if s.store else False
        if not fitted:
            s.store = s.load_normalize_store('KBins') if s.store_exists('KBins') else {}
        fitted = col_name in s.store.keys()
        if not fitted:
            est_name = s.pp_kbin_transform((np.random.choice(nda.flatten(), size=min(len(nda), sample_size), replace=False), col_name), n_bins=n_bins)
            s.store.update({est_name[1]: est_name[0]})
            s.store_normalized('KBins')
        return s.store[col_name].transform(s.nda1d_to_2d(nda)).astype(np.int8)

    @staticmethod
    def nda1d_to_2d(nda: np.ndarray) -> np.ndarray:
        return nda.reshape(-1, 1) if nda.ndim == 1 else nda

    @staticmethod
    def arr_num_tup_gen(df, sample_size=100000):
        if type(df) == pd.DataFrame:
            for col in df.columns:
                # print(col)
                yield df[col].sample(min(len(df[col]), sample_size)).values, col
        else:
            raise TypeError('Not compatible with object yet. Add handling')

    @staticmethod
    def pp_kbin_transform(arr_name_tup: (np.array, str), n_bins=20) -> tuple:
        arr, name = arr_name_tup
        logger.info(f'KBin fit {name}')
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
        est.fit(arr.reshape(-1, 1) if arr.ndim == 1 else arr)
        return est, name

    # def normalize_states(s, states: nda_schema):
    #     try:
    #         for ix, name in enumerate(states.schema):
    #             # for key in s.rl_bin_dict.keys():
    #             if name in s.rl_bin_dict.keys():
    #                 states.nda[:, ix] = s.rl_bin_dict[name].transform(states.nda[:, ix].reshape(-1, 1)).reshape(1, -1)[0]
    #         return states
    #     except AttributeError:
    #         raise AttributeError('No normalization schema provided or loaded before.')

    # @staticmethod
    # def get_bin_edges_by_equal_pop_split(states: nda_schema, n_bins=30):
    #     bin_dict = Dotdict()
    #     for ix, name in enumerate(states.schema):
    #         print('Search bins for {} ...'.format(name))
    #         est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
    #         if states.nda.dtype.names:  # rec array
    #             est.fit(states.nda[name].reshape(-1, 1))
    #         else:
    #             est.fit(states.nda[:, ix].reshape(-1, 1))
    #         bin_dict[name] = est
    #     return bin_dict

    # def set_normalization(s, data_dct: nda_schema):
    #     if isinstance(data_dct, dict):
    #         s.rl_bin_dict = data_dct
    #     else:
    #         s.rl_bin_dict = s.get_bin_edges_by_equal_pop_split(data_dct)
    #         with open(os.path.join(s.params.ex_path, f'bins_{s.params.asset.lower()}.obj'), 'wb') as f:
    #             pickle.dump(dict(s.rl_bin_dict), f)
