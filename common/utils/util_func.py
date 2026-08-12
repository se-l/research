import copy
import itertools
import math
import datetime as dt
import xgboost as xgb
import lightgbm as lgb
import re
import csv
import os
import errno
import time
import datetime
import pandas as pd
import numpy as np
import pickle

from enum import Enum
from typing import Union
from collections import namedtuple
from functools import reduce, lru_cache
from sklearn.model_selection import KFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.metrics import log_loss  # accuracy_score recall_score, precision_score, f1_score
from decimal import Decimal
from hyperopt.pyll.base import Apply
from statsmodels.tsa.stattools import adfuller

from ..modules.assets import Assets
from ..modules.direction import Direction
from ..modules.dotdict import Dotdict
from ..modules.rec_dotdict import recDotDict
from shared.modules.logger import logger
from ..paths import Paths
from ..refdata.named_tuples import FnO, DistSettings
from ..save import load_features_json, get_features_key

SeriesTickType = namedtuple('SeriesTickType', ('type', 'resample_val', 'folder'))


def pipe(funcs: [tuple], *args, **kwargs):
    """Creates a pipeline of data processing steps, where the first function can be initialized with data.
    Refactor to more intuitive. Entry data in first function call, not at end."""

    def wrapper():
        data_out = funcs[0](*args, **kwargs)
        for func in funcs[1:]:
            if isinstance(func, tuple) and len(func) == 2:
                # doesnt handle **kwargs in this place, need another type check
                data_out = func[0](*to_list(data_out), *to_list(func[1]))
            elif isinstance(func, tuple) and len(func) == 3:
                data_out = func[0](*to_list(data_out), *to_list(func[1]), **func[2])
            else:
                data_out = func(*to_list(data_out))
        return data_out

    return wrapper


def save_test_result(filename=None, folder=r'../log', **kwargs):
    if filename is None:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        filename = st

    mode = 'a' if os.path.isfile(os.path.join(folder, filename)) else 'w'

    with open(os.path.join(folder, filename), mode=mode) as f:
        w = csv.DictWriter(f, kwargs.keys())
        w.writeheader()
        w.writerow(kwargs)


def create_feature_map(features):
    outfile = open(os.path.join(Paths.projectDir, 'model', 'xgb.fmap'), 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def pickle_away(obj, dir1, dir2=None, ex=None, file_n_start='tmp', protocol=4, ts=None):
    dir_ = os.path.join(dir1, dir2) if dir2 is not None else dir1
    if ex is not None:
        dir_ = os.path.join(dir_, str(ex))
    create_dir(dir_)
    if ts is not None:
        pickle.dump(obj, open(
            os.path.join(dir_, '{}_{}'.format(
                file_n_start,
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'wb'), protocol)
    else:
        pickle.dump(obj, open(
            os.path.join(dir_, file_n_start), 'wb'), protocol)


def log_benchmark(logger, y_train, y_test):
    logger.debug('Benchmark alwUp Train / Test: {} / {}'.format(
        sum(y_train) / len(y_train),
        sum(y_test) / len(y_test)
    ))
    return sum(y_train) / len(y_train)


def get_important_features(cumsum_threshold, dir1, dir2, required=None):
    required = required or ['close']
    folder = os.path.join(dir1, dir2)
    feat_imp_list = []
    for root, dirs, filenames in os.walk(folder):
        for file in filenames:
            feat_imp_list.append(pickle.load(open(os.path.join(folder, file), 'rb')))
    feat_imp_list_merged = []
    sel = []
    for feats in feat_imp_list:
        sel.append(feats.iloc[np.where(feats['fscore'].cumsum() < cumsum_threshold)[0].tolist(), :])
        feat_imp_list_merged = feat_imp_list_merged + feats['feature'].tolist()
    feat_imp_list_merged = list(set(feat_imp_list_merged))
    sel = pd.concat(sel, axis=0)['feature'].tolist()
    sel = list(set(sel + required))
    return sel, [x for x in feat_imp_list_merged if x not in sel]

def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path

def interpolate_ts(df, maxmin=10, cols=None):
    cols = cols or ['open', 'high', 'low', 'close']
    slices, leave = get_null_ranges(df, maxmin)
    for s in slices:
        df.loc[s[0]:s[1], cols] = df.loc[s[0]:s[1], cols].interpolate(method='linear')
    return df, leave, slices


def get_null_ranges(df, maxmin=10):
    interp = []
    leave = []
    inb = False
    start = start_n = 0
    for i in range(0, len(df)):
        if pd.isnull(df.iloc[i, 3]):
            if not inb:
                start = df.index[i - 1]
                start_n = df.index[i]
            inb = True
        else:
            if inb:
                end = df.index[i]
                end_n = df.index[i - 1]
                if end - start <= datetime.timedelta(minutes=maxmin):
                    interp.append((start, end))
                else:
                    leave.append((start_n, end_n))
            inb = False
    return interp, leave


def invoke_method(obj, methods: Union[str, list], default=None):
    if isinstance(methods, str) and hasattr(obj, methods):
        return obj.__getattribute__(methods)
    elif isinstance(methods, list):
        for m in methods:
            if hasattr(obj, m):
                return obj.__getattribute__(m)
    return default


def gen_params(ps, params):
    i = 1 if invoke_method(params, 'max_evals', 0) > 1 else 0
    estimator_params = Dotdict()
    for k, v in ps.items():
        if isinstance(v, list) and (len(v) > 1 and isinstance(v[1], Apply) or k == 'estimator_args'):
            estimator_params[k] = v[i]
        else:
            estimator_params[k] = v
    return estimator_params


def gen_params_old(ps, params):
    max_evals = [p for p in dir(params) if 'max_evals' in p]
    i = 0
    for p in max_evals:
        if params.estimator_mode.mode in p.lower() and params.estimator_mode.estimator in p.lower():
            i = 1 if params.__getattribute__(p) > 1 else 0
    estimator_params = Dotdict()
    for k, v in ps.items():
        if type(v) == list and len(v) == 2:
            estimator_params[k] = v[i]
        else:
            estimator_params[k] = v
    return estimator_params


def lucena_split(x, y=None, n_folds=1, test_ratio=3):
    x, y = indexable(x, y)
    n_samples = _num_samples(x)
    if n_folds > n_samples:
        raise ValueError(
            ("Cannot have number of folds ={0} greater"
             " than the number of samples: {1}.").format(n_folds,
                                                         n_samples))
    indices = np.arange(n_samples)
    batch_size = (n_samples // n_folds)
    test_s = batch_size // test_ratio
    train_s = batch_size - test_s
    test_starts = range(0, n_samples - batch_size + n_samples % n_folds, test_s)
    for z in test_starts:
        if z >= n_samples - batch_size - test_s:
            yield (indices[z:z + train_s],
                   indices[z + train_s:n_samples])
        else:
            yield (indices[z:z + train_s],
                   indices[z + train_s:z + batch_size])


def split_off_holdout(d, params):
    split_ix = math.ceil(len(d) * (1 - params.holdoutSize))
    if type(d) in [pd.DataFrame, pd.Series]:
        return d.iloc[:split_ix, :], d.iloc[split_ix:, :]
    else:
        return d[:split_ix, :], d[split_ix:, :]


def df_to_npa(df, def_type=None) -> np.ndarray:
    if type(df) not in [pd.DataFrame, pd.Series]:
        print("Warning: the input is not a pandas dataframe, but {}".format(type(df)))
        return df
    v = df.values
    if type(df) == pd.DataFrame:
        cols = df.columns
        types = [(cols[i], get_col_type(df[k].dtype.type)) for (i, k) in enumerate(cols)]
    elif type(df) == pd.Series:
        cols = [df.name]
        types = [(cols[i], get_col_type(df.dtype.type)) for (i, k) in enumerate(cols)]
    else:
        raise TypeError
    types = [(cols[i], def_type) for (i, k) in enumerate(cols)] if def_type is not None else types
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    if len(v.shape) == 1:
        for (i, k) in enumerate(z.dtype.names):
            z[k] = v
    else:
        for (i, k) in enumerate(z.dtype.names):
            z[k] = v[:, i]
    return z


def get_col_type(type_):
    if type_ in [np.datetime64, 'datetime64[ns]']:
        return 'datetime64[ns]'
    else:
        return type_


def shift5(arr, num, fill_value=(np.nan,)):
    return shift5struc(arr, num, fill_value)


def shift5struc(arr, num, fill_value=(np.nan, np.nan, np.nan, np.nan)):
    """preallocate empty array and assign slice by chrisaycock"""
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


def make_struct_nda(v: np.ndarray, cols, def_type=None) -> np.ndarray:
    if def_type is None:
        if len(v.shape) == 1:
            types = [(cols[i], get_col_type(v[i].dtype.type)) for (i, k) in enumerate(cols)]
        else:
            types = [(cols[i], get_col_type(v[:, i].dtype.type)) for (i, k) in enumerate(cols)]
    else:
        types = [(cols[i], def_type) for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    if len(v.shape) == 1:
        z[z.dtype.names[0]] = v
    else:
        for (i, k) in enumerate(z.dtype.names):
            z[k] = v[:, i]
    return z


def join_struct_arrays(arrays, def_type=None):
    newdtype = sum((a.dtype.descr for a in arrays), [])
    if def_type is not None:
        n = [(x[0], def_type) for x in newdtype]
        newdtype = n
    newrecarray = np.empty(len(arrays[0]), dtype=newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


def struct_arr_rm_field_name(a, rm_col):
    names = list(a.dtype.names)
    for name in rm_col:
        if name in names:
            names.remove(name)
    return a[names]


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def load_pickle(dir1, dir2='/', rel_path=None, protocol=4):
    if rel_path:
        src = os.path.join(dir1, rel_path)
        if protocol == 2:
            with open(src, "rb") as f:
                item = pickle.load(f, fix_imports=True, encoding='latin-1')
            return item
        else:
            with open(src, "rb") as f:
                item = pickle.load(f)
                return item
    elif dir2:
        src = os.path.join(dir1, dir2)
        return list_pickle_files(src)


def list_pickle_files(dir_):
    out = []
    for root, dirs, filenames in os.walk(dir_):
        for file in filenames:
            with open(os.path.join(dir_, file), "rb") as f:
                out.append(pickle.load(f))
        break
    return out, filenames


def reduce_arr_length(dx_lw, dx_lw_ohlc, params):
    if type(dx_lw) == np.ndarray:
        dx_lw = dx_lw[:-params.period_shift]
    else:
        dx_lw = dx_lw.iloc[:-params.period_shift, :]
    if type(dx_lw_ohlc) == np.ndarray:
        dx_lw_ohlc = dx_lw_ohlc[:-params.period_shift]
    else:
        dx_lw_ohlc = dx_lw_ohlc.iloc[:-params.period_shift, :]
    return dx_lw, dx_lw_ohlc


def get_y_ps_dwin(dx_lw_ohlc, dx_lw, params):
    a = split_off_binary_y(dx_lw_ohlc, period_shift=params.period_shift, delta_win=params.delta_win)
    x, ohlc = reduce_arr_length(dx_lw, dx_lw_ohlc, params)
    return a, x, ohlc


def get_y_delta(dx_lw_ohlc, dx_lw, params):
    if 'forecast' in dir(params) and params.forecast == 'close':
        a = delta_close(dx_lw_ohlc, period_shift=params.period_shift)
    else:
        a = max_delta_y(dx_lw_ohlc, period_shift=params.period_shift, delta_win=params.delta_win)
    x, ohlc = reduce_arr_length(dx_lw, dx_lw_ohlc, params)
    return a, x, ohlc


def get_y_mat(dx_lw_ohlc, dx_lw, settings, perc_delta=True):
    shifts = list(set([s['period_shift'] for s in settings]))
    delta_win_list = list(set([s['delta_win'] for s in settings]))
    max_shift = max(shifts)

    y_mat = np.empty((len(dx_lw_ohlc) - max_shift, 4, len(shifts)))
    for shift_ix in np.arange(len(shifts)):
        # turn list of arrays into a p(dY >/< thresh | frequency) within each market state
        y_mat[:, 0:2, shift_ix] = max_delta_y(dx_lw_ohlc, period_shift=shifts[shift_ix], delta_win=0, perc_delta=perc_delta)
        # close
        y_mat[:, 2, shift_ix] = np.divide(
            np.subtract(dx_lw_ohlc.iloc[shifts[shift_ix]:, dx_lw_ohlc.columns.get_loc('close')], dx_lw_ohlc.iloc[:-shifts[shift_ix], dx_lw_ohlc.columns.get_loc('close')]),
            dx_lw_ohlc.iloc[:-shifts[shift_ix], dx_lw_ohlc.columns.get_loc('close')])
        # y_mat[:, 2, shift_ix] = dx_lw_ohlc[shifts[shift_ix]:, 3]
        # linear weighting function giving recent sample 3 times more weight than old ones
        y_mat[:, 3, shift_ix] = [1 + 2 * x / len(dx_lw_ohlc) for x in list(np.arange(len(dx_lw_ohlc) - max_shift))]
        # split_off_binary_y(dx_lw_ohlc, period_shift=params.period_shift, delta_win=params.delta_win)

    dx_lw, dx_lw_ohlc = reduce_arr_length(dx_lw, dx_lw_ohlc, Dotdict({'period_shift': max_shift}))
    return y_mat, dx_lw, dx_lw_ohlc, shifts, delta_win_list


def get_y_search(ohlc, settings, perc_delta=True):
    shifts = list(set([s['period_shift'] for s in settings]))
    max_shift = max(shifts)
    n_dim_mhl = 4 if perc_delta else 2
    y = np.empty(shape=(len(ohlc) - max_shift, n_dim_mhl, len(shifts)))
    for shift_ix in np.arange(len(shifts)):
        # turn list of arrays into a p(dY >/< thresh | frequency) within each market state
        if perc_delta:
            y[:, 0:4, shift_ix] = max_delta_y(ohlc, period_shift=shifts[shift_ix], delta_win=0, perc_delta=perc_delta)[
                                  :len(y)]
        else:
            y[:, 0:2, shift_ix] = max_delta_y(ohlc, period_shift=shifts[shift_ix], delta_win=0, perc_delta=perc_delta)[:len(y)]
        print('Finished {}% - returned shift {}'.format(100 * shift_ix / max_shift, shifts[shift_ix]))
    return y


def split_off_binary_y(d, period_shift=1, delta_win=0):
    perc_delta = True  # if abs(delta_win) < 2 else False
    # label 1 if a delta_win/Loss (current close and future HIGH/LOW) is achieved
    # between current time and future time, determined by period_shift
    if type(d) == pd.DataFrame:
        ndas = df_to_npa(d)
    else:
        ndas = d

    perc_delta_f = ndas['close'][:-period_shift] if perc_delta else 1

    if delta_win > 0:
        strides_high = rolling_window(ndas['high'], window=period_shift + 1)
        maxhigh = np.subtract(np.max(strides_high, axis=1), delta_win * perc_delta_f)
        y = (maxhigh > ndas['close'][:-period_shift]).astype(int)
    elif delta_win < 0:
        strides_low = rolling_window(ndas['low'], window=period_shift + 1)
        minlow = np.subtract(np.min(strides_low, axis=1), delta_win * perc_delta_f)
        y = (minlow < ndas['close'][:-period_shift]).astype(int)
    else:
        ValueError('No valid delta_win has been passed')

    return y
    # return join_struct_arrays([
    #     ndas[:-(period_shift - 1)],
    #     make_struct_nda(y, cols=['y'])
    # ])


def delta_close(d, period_shift, perc_delta=True):
    if type(d) == pd.DataFrame:
        ndas = df_to_npa(d)
    else:
        ndas = d

    if perc_delta:
        return np.divide(
            np.subtract(
                ndas['close'][period_shift:], ndas['close'][:-period_shift]),
            ndas['close'][:-period_shift]
        )
    else:
        return np.subtract(
            ndas['close'][period_shift:],
            ndas['close'][:-period_shift])


def max_delta_y(d, period_shift=1, delta_win=0, perc_delta=True):
    # label 1 if a delta_win/Loss (current close and future HIGH/LOW) is achieved
    # between current time and future time, determined by period_shift
    # perc_delta = True  # if abs(delta_win) < 2 else False

    if type(d) == pd.DataFrame:
        ndas = df_to_npa(d)
    else:
        ndas = d

    if delta_win >= 0:
        strides_high = rolling_window(ndas['high'], window=period_shift + 1)
        maxhigh = np.max(strides_high, axis=1)
        if delta_win > 0:
            perc_delta_f = ndas['close'][:len(maxhigh)] if perc_delta else 1
            return np.divide((maxhigh - ndas['close'][:len(maxhigh)]), perc_delta_f)

    if delta_win <= 0:
        strides_low = rolling_window(ndas['low'], window=period_shift + 1)
        minlow = np.min(strides_low, axis=1)
        if delta_win < 0:
            perc_delta_f = ndas['close'][:len(minlow)] if perc_delta else 1
            return np.divide((minlow - ndas['close'][:len(minlow)]), perc_delta_f)

    if delta_win == 0:
        # check for both minlow and maxhigh
        n_dim_mhl = 4 if perc_delta else 2
        nda = np.empty((len(d) - period_shift, n_dim_mhl))

        if perc_delta:
            nda[:, 2] = minlow - ndas['close'][:len(nda)]
            nda[:, 3] = maxhigh - ndas['close'][:len(nda)]
            nda[:, 0] = nda[:, 2] / ndas['close'][:len(nda)]
            nda[:, 1] = nda[:, 3] / ndas['close'][:len(nda)]
        else:
            nda[:, 0] = minlow - ndas['close'][:len(nda)]
            nda[:, 1] = maxhigh - ndas['close'][:len(nda)]
        return nda[:, 0:2]
    else:
        raise ValueError('No valid y!')


def get_predict_error(y_test, preds_tup, metric=('logloss',
                                                 'accuracy',
                                                 'sum_positives',
                                                 'sum_negatives',
                                                 'recall_score',
                                                 'precision_score',
                                                 'f1_score',
                                                 'pred_pos',
                                                 'pred_neg',
                                                 'first_pred_pos',
                                                 'first_pred_neg',
                                                 'first_precision_score'),
                      pred_thresh=0.5, err=None, stats=None):
    err = err or {}
    preds_orig, pred_bnry_orig, pred_bnry_first_orig = preds_tup

    pred_bnry = pred_bnry_orig[:len(y_test)]
    if len(pred_bnry_first_orig) > 0:
        pred_bnry_first = pred_bnry_first_orig[pred_bnry_first_orig < len(y_test)]
    else:
        pred_bnry_first = []

    TP = np.sum(np.logical_and(pred_bnry == 1, y_test == 1))
    TN = np.sum(np.logical_and(pred_bnry == 0, y_test == 0))
    FP = np.sum(np.logical_and(pred_bnry == 1, y_test == 0))
    FN = np.sum(np.logical_and(pred_bnry == 0, y_test == 1))

    y_test_first = y_test[pred_bnry_first]
    TPf = np.sum(y_test_first)
    # TNf = np.sum(y_test_first[y_test_first == 0] == pred_bnry_first[y_test_first == 0])
    FPf = len(y_test_first) - TPf
    # FNf = np.sum(y_test_first[y_test_first == 0] == pred_bnry_first[y_test_first == 1])

    # specificity  = TN / (TN+FP)
    # pos_pred_val = TP / (TP+FP)
    # neg_pred_val = TN / (TN+FN)

    if stats is None:
        stats = Dotdict([
            ('sum_y_test', np.sum(y_test)),
            ('len_y_test', len(y_test)),
            ('sum_pred_bnry', np.sum(pred_bnry)),
            ('len_pred_bnry', len(pred_bnry)),
            ('len_pred_bnry_first', len(pred_bnry_first)),
        ])
    else:
        stats.sum_pred_bnry = np.sum(pred_bnry)
        stats.len_pred_bnry_first = len(pred_bnry_first)

    # print('get pred err Len of preds - bnry: {} - {}'.format(len(preds), len(pred_bnry)))
    if err == {}:
        if 'logloss' in metric:
            try:
                err['logloss'] = log_loss(y_test, preds_orig[:stats.len_y_test])
            except ValueError:
                err['logloss'] = .99
        if 'sum_positives' in metric:
            err['sum_positives'] = stats.sum_y_test
        if 'sum_negatives' in metric:
            err['sum_negatives'] = stats.len_y_test - stats.sum_y_test

    if 'accuracy' in metric:
        err['accuracy'] = (TP + TN) / stats.len_y_test  # accuracy_score(y_test, pred_bnry)
    if 'recall_score' in metric:
        err['recall_score'] = TP / (TP + FN)
        # err['recall_score'] = recall_score(y_test, pred_bnry)
    if 'precision_score' in metric:
        err['precision_score'] = TP / (TP + FP)
        # err['precision_score'] = precision_score(y_test, pred_bnry)
    if 'f1_score' in metric:
        err['f1_score'] = 2 * (err['precision_score'] * err['recall_score']) / (err['precision_score'] + err['recall_score'])
    # if 'f1_diff' in metric:      'f1_diff'
    #     err['f1_diff'] =
    if 'pred_pos' in metric:
        err['pred_pos'] = stats.sum_pred_bnry
    if 'pred_neg' in metric:
        err['pred_neg'] = stats.len_pred_bnry - stats.sum_pred_bnry
    if 'first_pred_pos' in metric:
        err['first_pred_pos'] = stats.len_pred_bnry_first
    if 'first_pred_neg' in metric:
        err['first_pred_neg'] = stats.len_pred_bnry - stats.len_pred_bnry_first
    if 'first_precision_score' in metric:
        err['first_precision_score'] = TPf / (TPf + FPf)  # precision_score(y_test[pred_bnry_first], [1] * stats.len_pred_bnry_first)
    return err, stats


def rolling_window(a, window, writeable=True, adjust_window_at_edge=True):
    """Pay attention not to creat a forward looking vector. The returned share of a 1D array of size n is n - window -1.
    Hence insert into dataframes like this df[out][:win-1] = rollowing(df[in], win)
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=writeable)


def round_to_x(num_vec, x):
    remaining = np.mod(num_vec, x)
    return np.where(np.less(remaining, x / 2), num_vec - remaining, num_vec + (x - remaining))


def dotdict_to_dict(dic):
    for k, v in dic.items():
        if type(dic[k]) == Dotdict or type(dic[k]) == recDotDict:
            dic[k] = dotdict_to_dict(dic[k])
    return dict(dic)


def gen_xy_with_linear_shift(dx_lw, dx_lw_ohlc, settings):
    # predict continuous variable for specified periods into the future
    # each period is appended as column
    target = 'close'
    shift = settings.period_shift
    dx_lw, dy_l = get_linear_y(dx_lw, dx_lw_ohlc, target=target, shift=shift)
    # dx_lw, dy_l = get_normed_linear_y(dx_lw, dx_lw_ohlc, target=target, shift=shift)
    return dx_lw, dy_l


def get_linear_y(d, d_ohlc, target, shift):
    y = d_ohlc[target].shift(-shift) - d_ohlc[target]
    return d.iloc[:-shift], y[:-shift]


def get_normed_linear_y(d, d_ohlc, target, shift):
    y = d_ohlc[target].shift(-shift) - d_ohlc[target]
    d_ohlc = d.iloc[:-shift], d_ohlc[:-shift]
    # return d.iloc[:-shift, :], y.iloc[:-shift].divide(d_ohlc[target])
    return np.divide(y[:-shift], d_ohlc[target])


def insert_nda_col(arr: np.ndarray, values=None):
    return np.concatenate(
        [arr,
         values if values else np.empty((arr.shape[0], 1))
         ], axis=1
    )


def to_list(x=None) -> list:
    return to_iterable(x, list)


def resolve_col_name(col: str) -> (str, str):
    resolved = tuple(col.split('.'))
    if len(resolved) > 2:
        raise NotImplementedError('Too many dots in column. Cannot resolve properly yet')
    return resolved if len(resolved) == 2 else (None, resolved[0])


def to_iterable(x=None, dtype=None):
    if x is None:
        return []
    elif isinstance(x, (list, type)) and x:
        return list(x)
    elif isinstance(x, (np.ndarray, pd.Series)):
        if dtype is list:
            return x.tolist()
        else:
            return x
    else:
        return [x]


def auto_reduce_bool(scores, params, feature_names, model=None):
    """"""
    # ('min_sel_features_iterations', 2),
    # ('min_sel_features_n_remove', 2),
    # ('min_sel_features', 3),
    # ('min_sel_steps', 1),
    # ('min_sel_early_stopping_rounds', 1),
    if len(scores) == 0:
        # means no tests have been made yet
        return True
    if model is not None:
        try:
            n_model_feats = len(model.feature_names)
        except AttributeError:
            n_model_feats = len(model.feature_name())
    else:
        return True
    if n_model_feats <= params.min_sel_features or len(feature_names) == n_model_feats:
        return False
    elif len(scores) < params.min_sel_early_stopping_rounds + 1:
        # havent tested enough yet
        return True
    elif scores[-1] < np.min(scores[-1 - params.min_sel_early_stopping_rounds: -1]):
        return True
    elif np.argmin(scores) >= (len(scores) - params.min_sel_early_stopping_rounds):
        # the minimum is within last stopping rounds, therefore can
        return True
    else:
        return False


def update_dic(tdic, ndic):
    for k, v in ndic.items():
        tdic[k] = v
    return tdic


def get_feature_names(d) -> list:
    if type(d) == pd.DataFrame:
        return list(d.columns)
    elif type(d) == np.ndarray:
        return list(d.dtype.names)
    else:
        return ['f{}'.format(i) for i in np.arange(0, len(d))]


def get_model_features(m):
    if type(m) == xgb.Booster:
        return m.feature_names
    elif type(m) == lgb.Booster:
        return m.feature_name()


def get_model_fscore(m, importance_type='gain', iteration=None) -> dict:
    if type(m) == xgb.Booster:
        temp = m.get_score(importance_type=importance_type)
        temp = pd.DataFrame([temp.keys(), temp.values()]).transpose()
        temp.columns = ['name', 'fscore']
        return dict(zip(list(temp['name']), list(temp['fscore'])))
    elif type(m) == lgb.Booster:
        return dict(zip(m.feature_name(), m.feature_importance(importance_type=importance_type, iteration=iteration).tolist()))


def get_log_evals_results(evr, i, split_i, algo_type):
    if algo_type == ('regr', 'lgb'):
        return [
            evr[i][split_i]['train']['l1'],
            evr[i][split_i]['valid_0']['l1']
        ]
    elif algo_type == ('regr', 'xgb'):
        return [
            evr[i][split_i]['train']['mae'],
            evr[i][split_i]['val']['mae'],
        ]
    elif algo_type == ('class', 'lgb'):
        return [
            evr[i][split_i]['train']['binary_logloss'],
            evr[i][split_i]['valid_0']['binary_logloss'],
        ]
    elif algo_type == ('class_exit', 'lgb'):
        return [
            evr[i][split_i]['train']['xentropy'],
            evr[i][split_i]['valid_0']['xentropy'],
        ]
    elif algo_type == ('class', 'xgb'):
        return [
            evr[i][split_i]['train']['logloss'],
            evr[i][split_i]['val']['logloss'],
        ]
    else:
        print('no evals type identified')
        return []


def add_ts(d, start_md=(1, 1)):
    d['ts'] = d.index
    d['ts'] = d['ts'] + (
            dt.datetime(2018, month=start_md[0], day=start_md[1]) -
            dt.datetime(1970, 1, 1)).total_seconds()
    d['ts'] = d['ts'].apply(
        lambda x: dt.datetime.utcfromtimestamp(x))
    d.index = d['ts']
    d.drop('ts', axis=1, inplace=True)
    return d


def separate_entry_range(entry, dist=30):
    """
    requires data to be in pd.dataframe.
    Also all entries must already be above required prediction threshold.
    :param entry:
    :param dist:
    :return:
    """

    if isinstance(entry, pd.DataFrame) and isinstance(entry.index, pd.core.indexes.datetimes.DatetimeIndex):
        td = np.timedelta64(dist, 's')
        t0 = np.datetime64('1970-01-01')
        ix_dist = np.subtract(entry.index,
                              shift5(entry.index, 1, 0)
                              ).values - t0
        return entry.iloc[np.where(ix_dist > td)[0], :], entry.iloc[np.where(ix_dist <= td)[0], :]

    else:
        entry2 = shift5(entry, 1, 0)
        ix_dist = np.subtract(entry, entry2)
        ix_dist[0] = dist + 1  # ensures first entry gets counted

    return entry[np.where(ix_dist > dist)], entry[np.where(ix_dist <= dist)]


def myti(fn, n=10):
    t = []
    for i in range(n):
        t0 = time.time()
        fn()
        t.append(time.time() - t0)
    print('Executed fn {} times: Total: {}, Mean t: {}'.format(n, np.sum(t), np.mean(t)))


def divby(dic, key, by=10000):
    try:
        dic[key] = dic[key] / by
    except KeyError:
        pass
    return dic


@lru_cache()
def ex(sym: Assets = None) -> str:
    return f'ex{datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")}-' \
           f'{sym if sym else ""}'


def set_ex(params):
    """'0' or None for fresh init"""
    if params.ex in ['0', None]:
        return 'ex{}-{}'.format(
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            params.asset.lower(),
        )
    else:
        try:
            return params.ex
        except AttributeError:
            return None


def standard_params_setup(params, ex_path_dir):
    params.data_start = params.ts_start - datetime.timedelta(days=1)
    params.data_end = params.ts_end + datetime.timedelta(days=1)
    params.ex = set_ex(params)
    params.ex_path = os.path.join(ex_path_dir, params.ex)
    create_dir(params.ex_path)
    logger.add_file_handler(os.path.join(ex_path_dir, params.ex, 'log_{}'.format(datetime.date.today())))
    logger.debug('DataPeriod: {} - {}'.format(params.data_start, params.data_end))


def precision_and_scale(x):
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return magnitude, 0
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    scale = int(math.log10(frac_digits))
    return magnitude + scale, scale


def add_settings_features(params):
    features_dct = load_features_json()
    try:
        return features_dct[get_features_key(params)]
    except KeyError:
        return []


def find_indicator_ready_ix(df_full):
    longest_tp = 0
    longest_k = None
    for k in df_full.columns:
        long = max([int(i) for i in re.findall(r'\d+', k)])
        if 'ULTOSC' in k:
            long = long // 10
        if long > longest_tp:
            longest_tp = long
            longest_k = k
    return next((i for i, x in enumerate(df_full[longest_k]) if not np.isnan(x)), None)  # != df_full[longest_k][0]), None)


def find_max_nan_len(df):
    ll = 0
    for col in df.columns:
        col_nan_len = next((i for i, x in enumerate(df[col]) if not np.isnan(x)), None)
        ll = col_nan_len if col_nan_len > ll else ll
    return ll


def set_up_ex(args, params, ex='0'):
    args.ex = set_ex(ex, params.asset)  # '0' or None for fresh init
    create_dir(os.path.join(Paths.trade_model, args.ex))
    logger.add_file_handler(os.path.join(Paths.trade_model, args.ex, 'log_{}'.format(datetime.date.today())))
    logger.debug('Arguments: {}'.format(args))
    logger.debug('DataPeriod: {} - {}'.format(params.data_start, params.data_end))
    return logger


def ndas_to_regr_tup_dic(ndas):
    regr = {}
    for k in ndas.dtype.names:
        if 'regr' in k:
            snip = k.split('_')
            algo = 'xgb' if 'xgb' in snip else 'lgb'
            # side = 'short' if 'dwin--' in snip else 'long'
            ps = int(re.search(r'(?<=tp-)\d+', k).group())
            regressed = [r for r in snip if r in ['close', 'minlow', 'maxhigh']][0]
            regr[(algo, regressed, ps)] = ndas[k]
        else:
            continue
    return regr


def default_to_py_type(o):
    if isinstance(o, np.int64):
        return int(o)
    elif isinstance(o, np.int32):
        return int(o)
    else:
        return o


def date_day_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def date_sec_range(start_date, end_date):
    for n in range(int((end_date - start_date).total_seconds())):
        yield start_date + datetime.timedelta(seconds=n)


def map_symbol(sym):
    if sym == 'xrpxbt':
        return 'xrpusd'
    else:
        return sym


def rev_map_symbol(sym):
    if sym == 'xrpusd':
        return 'xrpxbt'
    else:
        return sym


def invert_y(y):
    return -1 * np.array(y) + 2 * max(y)


def todec(x):
    if type(x) == pd.Series:
        return x.apply(lambda el: Decimal(str(el)))
    else:
        return Decimal(str(x))


def get_intersect_ts(*args: pd.DataFrame, assume_unique=False):
    intersect_ts = args[0].index
    for i in range(1, len(args)):
        try:
            intersect_ts = np.intersect1d(intersect_ts, args[i].index, assume_unique=assume_unique)
        except AttributeError:
            # may not deal with a dataframe, but list of timestamps
            intersect_ts = np.intersect1d(intersect_ts, args[i], assume_unique=assume_unique)
    return intersect_ts


def reduce_to_intersect_ts(*args: pd.DataFrame):
    intersect_ts = get_intersect_ts(*args)
    return (df.loc[intersect_ts] for df in args)


def load_file(snippet, dir_) -> [FnO]:
    unpickled = []
    for root, dirs, filenames in os.walk(dir_):
        for file in filenames:
            try:
                re.search(snippet, file).group()
                with open(os.path.join(dir_, file), "rb") as f:
                    unpickled.append(FnO(file, pickle.load(f)))
            except AttributeError:
                pass
        break
    return unpickled


def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def fill_ix_keep_gaps(tt, min_gap=10):
    tr = []
    for i in range(len(tt)):
        if i > 0:
            tr.append(tt[i - 1])
            if tt[i] - tt[i - 1] >= min_gap:
                for ix in range(tt[i - 1] + min_gap, tt[i], min_gap):
                    tr.append(ix)
    return tr


def remove_ohlc_wo_price_change_2(ohlc_mid):
    print('Removing entries without price change')
    ix_close = ohlc_mid.columns.get_loc('close')
    mask_keep = ohlc_mid.iloc[1:, ix_close].values == ohlc_mid.iloc[:-1, ix_close].values
    keep = np.where(~mask_keep)[0]
    keep = np.concatenate([keep, keep + 1])
    keep = np.unique(keep)
    keep.sort()
    keep = fill_ix_keep_gaps(keep, min_gap=10)
    print('Record cnt after removal: {}. Compressed to {}%'.format(len(keep), round(100 * len(keep) / len(ohlc_mid), 2)))
    return keep


def lob_ts_to_dt(lob_ts):
    return pd.to_datetime(lob_ts, format='%Y-%m-%dD%H:%M:%S.%f')


def filter_reduce(lst: iter, f_reduce, f_filter, starting_value=None):
    """
    group(list, sum, lambda x: x if x < 0 else None)
    :param lst: list of inputs
    :param f_reduce: function to aggregate a list. reduce / numpy / builtins
    :param f_filter: filter condition on elements
    :param starting_value:
    :return: E.g. group(-1, 0, 2, sum, lambda x: x if x < 0 else None)
    """
    if starting_value:
        return reduce(f_reduce, filter(f_filter, lst), starting_value)
    else:
        return reduce(f_reduce, filter(f_filter, lst), starting_value)


def count_non_zero(lst: iter):
    return filter_reduce(lst, lambda x, y: x + 1, lambda x: x != 0, 0)


def count_lt_zero(lst: iter):
    return filter_reduce(lst, lambda x, y: x + 1, lambda x: x < 0, 0)


def count_gt_zero(lst: iter):
    return filter_reduce(lst, lambda x, y: x + 1, lambda x: x > 0, 0)


def sum_gt_zero(lst: iter):
    return filter_reduce(lst, lambda x, y: x + y, lambda x: x > 0, 0)


def sum_lt_zero(lst: iter):
    return filter_reduce(lst, lambda x, y: x + y, lambda x: x < 0, 0)


def distribute_into_params(params, *args):
    for distribute_settings in args:
        for dist_name, val in distribute_settings.items():
            params.__setattr__(dist_name, val)
    return params


def iter_merge(params, distribute_settings: DistSettings):
    for dist_params in distribute_settings.settings:
        new_param = copy.deepcopy(params)
        if distribute_settings.name is None:
            for key, val in dist_params.items():
                new_param.__setattr__(key, val)
        else:
            new_param.__setattr__(distribute_settings.name, dist_params)
        yield new_param


def align_ts_with_quantconnect(pdf, resolution):
    pdf['ts'] += pd.Timedelta(seconds=resolution)
    return pdf


def kf_shuffle(d: object, k_init: int = 5, x_tv_inv: object = None) -> object:
    """np unique returns list or unique market states in order of appearance, therefore
        more frequent important ones are at the beginning while exotic ones at the end. shuffling
        is required to have exotic and frequent states represented in each training batch.

        in k5 shuffling, np.uniques' list is slized into k**2 chunks and reassembled in a manner that each of the
        5 training batches is made up of a start, 3 middle and 1 end slice of the total training set,
        the shuffling is therefore not random
        """
    k_sq = k_init ** 2
    split_i = 0
    reassemble_d = {k: None for k in range(k_sq)}
    reassemble_index = reassemble_d.copy()
    # reassemble_d = {}
    # reassemble_index = {}
    # for i in range(k):
    #     reassemble_d[i] = None
    #     reassemble_index[i] = None
    reassemble_order = []
    kf = KFold(n_splits=k_sq, random_state=0)
    for ix_train, ix_test in kf.split(d):
        target_i = int(np.floor(split_i / k_init) + split_i * k_init - (np.floor(split_i / k_init)) * k_sq)
        reassemble_order.append(target_i)
        reassemble_d[target_i] = d[ix_test]
        reassemble_index[target_i] = ix_test
        split_i += 1
    f_stack = 'vstack' if len(d.shape) > 1 else 'hstack'
    d_rearranged = np.__getattribute__(f_stack)([reassemble_d[i] for i in reassemble_order])
    index_map = dict(zip(np.array(range(len(d))), np.hstack([reassemble_index[i] for i in reassemble_order])))
    # for unit testing
    # d= np.array(list(range(25)))
    # d, inv_map = kf_shuffle(d)
    # print(d)
    # print(inv_map)
    # ta=x_tv_uni[x_tv_inv]
    # tb = x_tv_uni[x_tv_inv]
    inv = list(index_map.values())
    # the product of the shape must equal the sum of true elements when comparing original array with the inverted rearranged array
    if not isinstance(d_rearranged, np.ndarray):
        assert np.sum(d == d_rearranged[inv]) + np.isnan(d).sum() == (reduce(lambda x, y: x * y, d.shape, 1)), 'no'
    # if x_tv_inv is not None:
    #     x_tv_inv2 = np.array([index_map[k] for k in x_tv_inv])
    #     assert np.sum(d[x_tv_inv]) == np.sum(d_rearranged[inv][x_tv_inv]), 'no'
    # assert np.sum(d[x_tv_inv]) == np.sum(d_rearranged[x_tv_inv2]), 'no'
    return d_rearranged, index_map


def kf_shuffle_np_uni(x_tv_uni, x_tv_inv, x_tv_uni_cnt, k=5):
    x_tv_uni, index_map = kf_shuffle(x_tv_uni, k, x_tv_inv)
    x_tv_uni_cnt, index_map = kf_shuffle(x_tv_uni_cnt, k)
    x_tv_inv = np.array([index_map[k] for k in x_tv_inv])
    return x_tv_uni, x_tv_inv, x_tv_uni_cnt


def convert_enum2str(key) -> Union[str, list]:
    if isinstance(key, list):
        key = [k.name if isinstance(k, Enum) else k for k in key]
    else:
        key = key.name if isinstance(key, Enum) else key
    return key


def downside_deviation(arr: np.array):
    sub = np.diff(arr)
    return np.where(sub <= 0, sub, 0).std()


def downside_deviation_rolling(arr: np.array) -> np.ndarray:
    sub = np.diff(arr)
    sub = np.where(sub <= 0, sub, 0)
    return np.array([0] + pd.DataFrame(sub).expanding(1).std(ddof=0)[0].values.tolist())


def predict_models(models_dct: dict, target, model=None, add_to=None, predictor=None):
    """model should be a dict: key: name of preds. val: model
    This is messy and should be turned into more functions as this can be called in multiple ways
    """
    if type(target) == str:
        with open(target, "rb") as f:
            target = pickle.load(f)

    if model is None:
        preds_col = []
        for k, v in models_dct.items():
            if type(v) == dict:
                preds_col += list(v.keys())
            else:
                preds_col.append(k)
        preds = np.full(shape=(target.shape[0],), fill_value=np.nan, dtype=[(name, np.float) for name in preds_col])
        # needs to go into multiprocessing
        for name, m in models_dct.items():
            if type(m) == dict:
                for name_rd, m_rd in m.items():
                    preds[name_rd] = predictor.predict(model=m_rd, target=target)
            # elif type(m) == list:
            #     for i in range(len(m)):
            #         preds[i] = s.predict(model=m[i], target=target)
            else:
                preds[name] = predictor.predict(model=m, target=target)
    else:
        # with Pool(processes=min(multiprocessing.cpu_count() // 2, len(model.keys()))) as p:
        #     preds = p.map(partial(predictor.predict, target=target), model.values())
        #     preds = make_struct_nda(np.array(preds), list(model.keys()))
        preds = np.full(shape=(target.shape[0],), fill_value=np.nan, dtype=[(name, np.float) for name in model.keys()])
        for name, m in model.items():
            preds[name] = predictor.predict(model=m, target=target)
    if add_to is not None:
        return join_struct_arrays([add_to, preds])
    else:
        return preds


def average_batch_models(preds, models_dct):
    avg_preds = np.full(shape=(preds.shape[0],), fill_value=np.nan,
                        dtype=[(name, np.float) for name in models_dct.keys()])
    batch_preds = list(itertools.chain(*[v.keys() for v in models_dct.values()]))
    non_batch_preds = [n for n in get_feature_names(preds) if n not in batch_preds]
    for k, v in models_dct.items():
        avg_preds[k] = np.mean(pd.DataFrame(preds[list(v.keys())]), axis=1)
    if len(preds[non_batch_preds]) > 0:
        return join_struct_arrays([preds[non_batch_preds], avg_preds])
    else:
        return avg_preds


def full_predict_uni(models_dct: dict, target, predictor) -> np.ndarray:
    """this is for making predictions that can be used in backtesting. Not for stacking as we're predicting
    values on trained data. This method is not for training"""
    x_tv_uni, x_tv_inv, x_tv_uni_cnt = np.unique(target, axis=0, return_inverse=True, return_counts=True)
    if 'void' not in x_tv_uni.dtype.name:
        x_tv_uni = make_struct_nda(x_tv_uni, get_feature_names(target))
    all_models = {k: v for k, v in [item for sublist in models_dct.values() for item in sublist.items()]}
    non_avg_preds = np.full(shape=(x_tv_uni.shape[0],), fill_value=np.nan,
                            dtype=[(name, np.float) for name in all_models.keys()])
    for m_n, m in all_models.items():
        non_avg_preds[m_n] = predict_models(models_dct, x_tv_uni, model={m_n: m}, predictor=predictor)
    preds = average_batch_models(non_avg_preds, models_dct)
    preds = preds[x_tv_inv]
    assert len(preds) == len(target), 'Length of preds not equal to final target. mismatch'
    return preds


def pdf_col_ix(pdf, name):
    return pdf.columns.get_loc(name)


def total_profit2(order_lst):
    return [sum([(order.fill.avg_price - order_lst[i - 1].fill.avg_price) * (-1 if order.direction == Direction.long else 1) for i, order in enumerate(order_lst[:i]) if i % 2 != 0]) for i in
            range(len(order_lst))]


def is_stationary(arr: np.array, threshold: float = 0.05, first_x_pct=0.05) -> bool:
    """
    Augmented Dickey-Fuller test
    https://machinelearningmastery.com/time-series-data-stationary-python/
    p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

    ADF Test requires OLS on ps[::lag:]. time consuming => Sample it for speed.
    """
    try:
        ix_end = int(len(arr) * first_x_pct) if len(arr) > (first_x_pct * 100_000) else len(arr)
        result = adfuller(arr[:ix_end])
        logger.info(f'''p-value: {result[1]} - ADF Statistic: {result[0]}''')
        # for key, value in result[4].items():
        #     print('\t%s: %.3f' % (key, value))
        return result[1] <= threshold
    except Exception as e:
        logger.warning(e)
        return False
