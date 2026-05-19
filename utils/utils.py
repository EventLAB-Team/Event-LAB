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
    logger.info('Version 1.0.0 - May 2026')
    logger.info('')