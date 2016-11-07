# coding: utf-8

from argparse import ArgumentParser
import json
from logging import basicConfig, getLogger

import yaml


def load_config(config_file):
    if config_file.endswith('.yml') or config_file.endswith('.yaml'):
        with open(config_file) as f:
            return yaml.load(f)
    else:
        raise ValueError('{} is unsupported'.format(config_file))


def dict_merge(x, y):
    merged = dict(x, **y)

    xkeys = x.keys()
    for key in xkeys:
        if isinstance(x[key], dict) and key in y:
            merged[key] = dict_merge(x[key], y[key])

    return merged


def invoke_function(default_config, local_config):
    function_name = local_config.pop('function')
    if function_name is None:
        raise ValueError('function is not defined.')
    names = function_name.split('.')
    command_func = 'import json\n'
    learner_config = {}
    if default_config is not None:
        learner_config.update(default_config)
    learner_config = dict_merge(learner_config, local_config)
    command_func += 'learner_config = json.loads("' + json.dumps(learner_config).replace('\\', '\\\\').replace('"', '\\"') + '")\n'
    if len(names) == 1:
        command_func += 'import ' + names[0] + '\n' + names[0] + '(learner_config)'
    else:
        command_func += 'from ' + '.'.join(names[:-1]) + ' import ' + names[-1] + '\n' + names[-1] + '(learner_config)'
    exec(command_func)


def run(config_file='config.yml'):
    parser = ArgumentParser(description='Learner Framework invokes Machine Learning actions.')
    parser.add_argument('command', default='default')
    parser.add_argument('--config', dest='config_file', default=config_file)
    options = parser.parse_args()

    config = load_config(options.config_file)

    logging_config = config.get('logging')
    if logging_config is not None:
        basic_logging_config = logging_config.pop('basic_config')
        if basic_logging_config is not None:
            basicConfig(**basic_logging_config)
        for k, v in logging_config.items():
            logger = getLogger(k)
            if 'level' in v:
                logger.setLevel(v.get('level'))

    if options.command in config:
        return invoke_function(config.get('default'), config.get(options.command))
    else:
        raise ValueError('{} is not found.'.format(options.command))
