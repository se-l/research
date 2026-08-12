import os
import pickle
import json

from functools import reduce
from ast import literal_eval
from collections import namedtuple
from common.paths import Paths

json_key_templates = {
    'model_train_features': [
        'asset',
        'direction',
        'estimator_mode.mode',
        'estimator_mode.estimator',
        'series_tick_type.type',
        'feature',
        'discount_decay',
        'series_tick_type.resample_val',
        'exchange'
    ]
}

save_defaults = {
    'model_train_features': {
        'pattern': namedtuple('ModelTrainFeatures', ''),
        'location': Paths.model_features,
        'file_type': 'json'
    },
    'class_model_stacks': {'pattern': namedtuple('ModelStacks', 'mode est side ts')},
    'regr_model_stacks': {'pattern': namedtuple('ModelStacks', 'mode est side ts')}
}

file_type_f = {
    'json': json.dumps,
    'p': pickle.dump
}


def nested_attr(obj, attr: str):
    attr_lst = attr.split('.')
    ga = lambda obj, el: obj.__getattribute__(el)
    return reduce(lambda x, y: ga(x, y), attr_lst, obj)


def get_features_key(params):
    return tuple([str(nested_attr(params, attr)) for attr in json_key_templates['model_train_features']])


def save_features_json(obj, params, path=None):
    try:
        with open(path or os.path.join(Paths.model_features), 'rt') as f:
            feats_dct = json.load(f)
    except FileNotFoundError:
        feats_dct = {}
    feats_dct[str(get_features_key(params))] = obj
    with open(path or os.path.join(Paths.model_features), 'wt') as f:
        json.dump(feats_dct, f)


def load_features_json():
    try:
        with open(os.path.join(Paths.model_features), 'rt') as f:
            return {literal_eval(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}


def save(obj, params, file_type=None, location=None):
    if isinstance(params, dict):
        fn = str(SaveName(params)) + '.' + params['file_type']
    else:
        raise NotImplementedError


class SaveName:
    def __init__(self, pattern: object):
        self.pattern: dict = pattern._asdict() if isinstance(pattern, tuple) else pattern

    def __str__(self):
        return '|'.join(['_'.join([str(k), str(v)]) for k, v in self.pattern.items()])
