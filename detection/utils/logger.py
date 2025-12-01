# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import torch
import torch.distributed as dist

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("Cannot import tensorboard. Will log to txt files only.")
    SummaryWriter = None

from utils.dist import is_primary


class Logger(object):
    def __init__(self, log_dir=None) -> None:
        self.log_dir = log_dir
        if SummaryWriter is not None and is_primary():
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

    def log_scalars(self, scalar_dict, step, prefix=None):
        if self.writer is None:
            return
        for k in scalar_dict:
            v = scalar_dict[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            if prefix is not None:
                k = prefix + k
            self.writer.add_scalar(k, v, step)

logger_initialized = {}

def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0


    if rank == 0 and log_file is not None:

        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True


    return logger

def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')