# import os
# import logging

# import torch.distributed as dist

# class RankFilter(logging.Filter):
#     def __init__(self, rank):
#         super().__init__()
#         self.rank = rank

#     def filter(self, record):
#         return dist.get_rank() == self.rank

# def create_logger(log_path):
#     # Create log path
#     if os.path.isdir(os.path.dirname(log_path)):
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)

#     # Create logger object
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     # Create file handler and set the formatter
#     fh = logging.FileHandler(log_path)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)

#     # Add the file handler to the logger
#     logger.addHandler(fh)

#     # Add a stream handler to print to console
#     # sh = logging.StreamHandler() 
#     # sh.setLevel(logging.INFO)  # Set logging level for stream handler
#     # sh.setFormatter(formatter)
#     # logger.addHandler(sh)

#     # Ensure no stream handler is added to the logger (do not print to console)
#     for handler in logger.handlers:
#         if isinstance(handler, logging.StreamHandler):
#             logger.removeHandler(handler)

#     return logger
# --------------------------------------------------------------------------- #
# taken from DFB github repo -> https://github.com/SCLBD/DeepfakeBench
import os
import logging
import shutil

import torch.distributed as dist # torch.distributed is a package that enables multi-process communication

class RankFilter(logging.Filter): # logging.Filter is a class that allows you to filter log records based on certain criteria (e.g. log level, message, etc.)
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return dist.get_rank() == self.rank

def create_logger(log_path):
    # Create log directory if it does not exist
    if os.path.isdir(os.path.dirname(log_path)):
        # if os.path.isfile(log_path): # if the log file already exists, delete it
        #     # os.remove(log_path)
        #     log_path = log_path.replace('.log', '_v2.log')
        #     print(log_path)
        #     os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_path = check_if_log_file_exists(log_path)
        print(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    else:
        # If the directory does not exist, create it
        os.makedirs(os.path.dirname(log_path))

    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    # asctime: human-readable time when the LogRecord was created
    # levelname: the log level of the LogRecord (e.g. INFO, WARNING, etc.)
    # message: the log message
    fh.setFormatter(formatter)

    # Add the file handler to the logger (write to file)
    logger.addHandler(fh) 

    # Add a stream handler to print to console
    # sh = logging.StreamHandler()
    # sh.setLevel(logging.INFO)  # Set logging level for stream handler
    # sh.setFormatter(formatter)
    # logger.addHandler(sh) # Add the stream handler to the logger (print to console)

    # Ensure no stream handler is added to the logger (do not print to console)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    return logger

def check_if_log_file_exists(log_path):
    if os.path.exists(log_path):
        if '_v' not in log_path:
            # log_path = log_path.replace('.log', '_v2.log')
            # return log_path
            version = 1
        else: version = 2
        # add "version 2" to the log file name
        
        while os.path.exists(log_path): # check if the log file exists in the directory
            if version == 1:
                log_path = log_path.replace('.log', f'_v{version}.log')
            log_path = log_path.replace(f'_v{version-1}.log', f'_v{version}.log')
            version += 1
        return log_path
    else:
        return log_path