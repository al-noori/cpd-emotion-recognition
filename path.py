from pathlib import Path
from socket import gethostname


"""
The following functions select paths based on the hostname of the system they are run on,
so that paths locally and on the server are automatically adjusted.
"""



if gethostname() in ['']:  # local: fill in your hostname here
    DATA_PATH = Path('..','data')
    MODEL_PATH = Path('..','model')
    PLOTS_PATH = Path('..', 'plots')
else:  # server
    DATA_PATH = Path('/work', '089888', 'data')
    PLOTS_PATH = Path('/work', '089888', 'plots')
    MODEL_PATH = Path('/home', 'users', '0015', 'uk089888', 'ComTech1', 'model')
