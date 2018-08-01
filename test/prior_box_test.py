import sys
sys.path.append('.')

from src.prior_box import PriorBox, PriorBoxV1
from src.config import config

cfg = config.list['v6']
priorbox = PriorBox(cfg)
prior_data = priorbox.forward()
print(prior_data)


cfg = config.list['v-test']
priorbox_v1 = PriorBoxV1(cfg)
prior_data_v1 = priorbox_v1.forward()
print(prior_data_v1)
