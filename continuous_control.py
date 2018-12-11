import numpy as np
import torch
import click
from unityagents import UnityEnvironment


@click.command()
@click.option(
    "--version",
    default=1,
    help   = """
        Select version of environment to run 
        
        1: single agent 
        
        2: multi agent
        
        """
)
@click.option(
    "--env-dir",
    default = "environments",
    help    = """
        Set directory containing environment(s)
    """
)
def run(version, env_dir):
    env = load_env(env_dir, version)

def load_env(env_dir, version):
    print('Loading environment at {}/Reacher_v{}\n'.format(env_dir, version))
    return UnityEnvironment(file_name='{}/{}'.format(env_dir, 'Reacher_v{}'.format(version)))

if __name__ == '__main__':
    run()
