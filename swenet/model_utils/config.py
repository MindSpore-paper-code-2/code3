# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Parse arguments"""
import os
import ast
from pprint import pprint, pformat
import argparse
import yaml


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Config(val) if isinstance(val, dict) else val)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_cli_to_yaml(parser, configs, helper=None, choices=None, cfg_path="default_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])

    for item in configs:
        if not isinstance(configs[item], list) and not isinstance(configs[item], dict):
            choice = choices[item] if item in choices else None
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)

            if isinstance(configs[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=configs[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + item, type=type(configs[item]), default=configs[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()

    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            configs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            configs = [x for x in configs]
            if len(configs) == 1:
                cfg_helper = {}
                cfg = configs[0]
                cfg_choices = {}
            elif len(configs) == 2:
                cfg, cfg_helper = configs
                cfg_choices = {}
            elif len(configs) == 3:
                cfg, cfg_helper, cfg_choices = configs
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
            print(cfg_helper)
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)

    for item in args_var:
        cfg[item] = args_var[item]

    return cfg


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="config parser", add_help=False)

    parser.add_argument("--config_path", type=str, default=os.path.join(current_dir, "../config.yaml"),
                        help="Config file path")
    path_argparse, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_argparse.config_path)
    args = parse_cli_to_yaml(parser=parser, configs=default, helper=helper,
                             choices=choices, cfg_path=path_argparse.config_path)
    final_config = merge(args, default)
    pprint(final_config)
    print("Please check the above information for the configurations", flush=True)
    return Config(final_config)

config = get_config()
