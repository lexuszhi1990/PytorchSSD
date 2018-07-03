import sys
sys.path.append('.')

from src.prior_box import PriorBox
from src.config import config

cfg = config.list['r1']

priorbox = PriorBox(cfg)
prior_data = priorbox.forward()
print(prior_data)
