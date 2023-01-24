import argparse
import json
import os
from typing import Callable, Dict

import yaml
from yaml import safe_load as yload

import setup_datasets
    

# pass name in as arg + config file
# only to add now, change to update existing later
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, help="path to (yaml) config file")
parser.add_argument("--dataset", type=str, nargs="+", required=True, help="1 or more dataset names as defined in cfg")
if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.cfg, 'r') as file:
        config = yload(file)

    for d in args.dataset:
        print(f'Adding dataset {d}')
        addfunc = getattr(setup_datasets, config['datasets'][d])
        config = addfunc(config)

    with open(args.cfg, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)