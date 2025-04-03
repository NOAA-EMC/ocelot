#!/usr/bin/env python3

import os
import pathlib
import shutil

configs_path = os.path.join(pathlib.Path(__file__).parent.resolve(), './')
settings_path = os.path.join(configs_path, 'settings.py')

if not os.path.exists(settings_path):
    template_path = os.path.join(configs_path, 'template_settings.py')
    shutil.copyfile(template_path, settings_path)

os.system(f"vi {settings_path}")
