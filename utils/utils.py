import importlib
from typing import Any


def get_class_from_str(string, reload=False):
    if string == 'None':
        def noop(*args, **kwargs):
            pass
        return noop

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def make_model_from_config(config) -> Any:
    return get_class_from_str(config.target)(**config.params)