# -*- coding: utf-8 -*-

from .log import setup_logger

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
