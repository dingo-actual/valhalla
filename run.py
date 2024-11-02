from src.apps.test_functions import test_main

from argparse import ArgumentParser
from json import load
from os.path import relpath, dirname, join
import sys


parser = ArgumentParser(
    prog='valhalla',
    description='population-based optimization'
)
subparsers = parser.add_subparsers(dest='app')

parser_test = subparsers.add_parser(
    'test', 
    help='test basic PSO on sphere function'
)
parser_test.add_argument(
    '--test', 
    action='store', 
    nargs=1, 
    type=int,
    required=True,
    choices=['sphere', 'rast'],
    help='benchmark function to test against'
)
parser_test.add_argument(
    '--opt', 
    dest='optimizer', 
    action='store', 
    nargs=1, 
    type=int,
    required=True,
    choices=['pso', 'acd_pso', 'ese_apso'],
    help='optimization algorithm'
)
parser_test.add_argument(
    '--dim', 
    dest='dim', 
    action='store', 
    nargs=1, 
    type=int,
    required=True,
    help='dimension of problem space'
)
parser_test.add_argument(
    '--n-part', 
    dest='n_particles', 
    action='store', 
    nargs=1, 
    type=int,
    required=True,
    help='number of particles'
)
parser_test.add_argument(
    '--n-iter', 
    dest='n_iter', 
    action='store', 
    nargs=1, 
    type=int,
    required=True,
    help='number of optimization iterations'
)
parser_test.add_argument(
    '--verbosity', 
    dest='verbosity', 
    action='store', 
    nargs=1, 
    type=int,
    required=True,
    help='output verbosity'
)
parser_test.add_argument(
    '--debug', 
    action='count', 
    help='output debug messages'
)


def main(args):
    app = args['app']
    
    if app == 'test':
        test_main(args)
    else:
        raise ValueError(f'unrecognized app name "{app}"')


if __name__=='__main__':
    gettrace = getattr(sys, 'gettrace')
    if gettrace is not None and gettrace():
        script_dir = dirname(relpath(__file__))
        debug_args_fpath = join(script_dir, 'debug_args.json')
        
        with open(debug_args_fpath, 'r') as fp:
            args = load(fp)
    else:
        args = dict(vars(parser.parse_args()).items())
        
    main(args)
