from dataclasses import fields
from typing import Dict


class DCBase:
    @classmethod
    def from_kw(cls, kw: Dict, k_map=None):
        return cls(**{k_map.get(k, k): v for k, v in kw.items() if k_map.get(k, k) in [f.name for f in fields(cls)]})

    @classmethod
    def from_obj(cls, obj: object, k_map=None):
        return cls(**{k_map.get(k, k): getattr(obj, k) for k in dir(obj) if k_map.get(k, k) in [f.name for f in fields(cls)]})
