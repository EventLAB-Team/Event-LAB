import time, os, sys
from tqdm import tqdm
from loguru import logger

def start(method, dataset):
    logger.remove()
    # Add the log file
    logpath = os.path.join("logs", dataset, method)
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(f"{logpath}/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{method}_{dataset}.log")

    logger.info('')
    logger.info('███████╗██╗   ██╗███████╗███╗   ██╗████████╗   ██╗      █████╗ ██████╗') 
    logger.info('██╔════╝██║   ██║██╔════╝████╗  ██║╚══██╔══╝   ██║     ██╔══██╗██╔══██╗')
    logger.info('█████╗  ██║   ██║█████╗  ██╔██╗ ██║   ██║█████╗██║     ███████║██████╔╝')
    logger.info('██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║╚██╗██║   ██║╚════╝██║     ██╔══██║██╔══██╗')
    logger.info('███████╗ ╚████╔╝ ███████╗██║ ╚████║   ██║      ███████╗██║  ██║██████╔╝')
    logger.info('╚══════╝  ╚═══╝  ╚══════╝╚═╝  ╚═══╝   ╚═╝      ╚══════╝╚═╝  ╚═╝╚═════╝ ')
    logger.info('')
    logger.info('Towards Standardized Evaluation of Neuromorphic Localization Methods')
    logger.info('================================================')
    logger.info('Adam D Hines*, Alejandro Fontan, Michael Milford, Tobias Fischer')
    logger.info('QUT Centre for Robotics, Queensland Australia')
    logger.info('')
    logger.info('*adam.hines@qut.edu.au')
    logger.info('================================================')
    logger.info('Version 1.1.0 - July 2026')
    logger.info('')

def convert_offset(offsetref, offsetqry, scale):
    # Check the offset timescale, convert to msec
    if scale == "s":
        return int(offsetref * 1000), int(offsetqry * 1000)
    elif scale == "us":
        return int(offsetref * 1000000), int(offsetqry * 1000000)
    elif scale == "ns":
        return int(offsetref * 1000000000), int(offsetqry * 1000000000)
    else:
        return int(offsetref), int(offsetqry)      

         

def pixi_run(environment, command, cwd=None, extra_env=None):
    """
    Run `command` inside a named Event-LAB pixi environment.

    Baselines whose dependencies conflict with the default environment declare a
    `[feature.<name>]` in pixi.toml and are invoked through here. Because this
    nests a `pixi run` inside the one that started eventlab_run.py, the inherited
    PIXI_ENVIRONMENT/PIXI_PROJECT_MANIFEST are stripped first -- otherwise the
    child can resolve straight back to the parent's environment.

    `cwd` is applied by prefixing `cd` to the command rather than by moving the
    subprocess. Several cloned baselines are pixi projects in their own right, so
    launching pixi from inside one makes it resolve `-e <environment>` against
    THAT manifest and fail with "unknown environment". pixi therefore always runs
    from the Event-LAB root, and only the inner command changes directory.
    """
    import shlex
    import subprocess

    env = os.environ.copy()
    env.pop("PIXI_ENVIRONMENT", None)
    env.pop("PIXI_PROJECT_MANIFEST", None)
    env.pop("PIXI_IN_SHELL", None)
    if extra_env:
        env.update(extra_env)

    if cwd:
        command = f"cd {shlex.quote(str(cwd))} && {command}"

    full = ["pixi", "run", "-e", environment, "bash", "-c", command]
    logger.info(f"[{environment}] {command}")
    return subprocess.run(full, env=env, text=True)
