import importlib.util

_numpy_available = importlib.util.find_spec("numpy") is not None
