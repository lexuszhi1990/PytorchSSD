from itertools import product as product
from math import sqrt as sqrt

import torch

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # center of default box
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if self.max_sizes:
                    s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class PriorBoxV1(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.
    Here we support arbitrary size of image.
    """

    def __init__(self, cfg):
        super(PriorBoxV1, self).__init__()
        self.image_size = cfg['shape']
        self.feature_maps = cfg['feature_maps']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                cx = (i + 0.5) / (self.image_size[0] / self.steps[k][0])
                cy = (j + 0.5) / (self.image_size[1] / self.steps[k][1])

                s_k_w = self.min_sizes[k][0] / self.image_size[0]
                s_k_h = self.min_sizes[k][1] / self.image_size[1]
                mean += [cx, cy, s_k_w, s_k_h]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if self.max_sizes:
                    s_k_w_prime = sqrt(s_k_w * (self.max_sizes[k][0] / self.image_size[0]))
                    s_k_h_prime = sqrt(s_k_h * (self.max_sizes[k][1] / self.image_size[1]))
                    mean += [cx, cy, s_k_w_prime, s_k_h_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k_w * sqrt(ar), s_k_h / sqrt(ar)]
                    mean += [cx, cy, s_k_w / sqrt(ar), s_k_h * sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
