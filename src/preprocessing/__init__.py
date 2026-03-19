from . import fft
from . import hermite_cubic_derivative
from . import domain_normalize

import inspect
import sys
from merlion.transform.base import TransformBase
from merlion.transform import factory as _merlion_factory

def _register_custom_transforms():
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("preprocessing"):
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, TransformBase) and cls is not TransformBase:
                _merlion_factory.import_alias[cls.__name__] = f"{cls.__module__}:{cls.__name__}"

_register_custom_transforms()