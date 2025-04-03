import os
import sys
import pathlib

sys.path.append(os.path.realpath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../configs')))

try:
    from local_settings import *  # noqa: F401, F403
except ImportError:
    raise ImportError("configs/settings.py was not found!.")
finally:
    sys.path.pop(0)
