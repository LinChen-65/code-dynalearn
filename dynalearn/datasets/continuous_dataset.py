import numpy as np
import networkx as nx
import torch
import tqdm

from scipy.stats import gaussian_kde
from dynalearn.datasets import Dataset, StructureWeightDataset
from dynalearn.datasets.weights import (
    ContinuousStateWeight,
    ContinuousGlobalStateWeight,
    ContinuousCompoundStateWeight,
    StrengthContinuousGlobalStateWeight,
    StrengthContinuousStateWeight,
    StrengthContinuousCompoundStateWeight,
)
from dynalearn.config import Config
from dynalearn.util import from_nary
from dynalearn.util import to_edge_index, onehot, get_node_attr

import pdb

class ContinuousDataset(Dataset):
    def __getitem__(self, index):
        i, j = self.indices[index]
        g = self.networks[i].get()
        x = torch.FloatTensor(self.inputs[i].get(j))
        y = torch.FloatTensor(self.targets[i].get(j))
        w = torch.FloatTensor(self.weights[i].get(j))
        w /= w.sum()
        return (x, g), y, w


class ContinuousStructureWeightDataset(ContinuousDataset, StructureWeightDataset):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        ContinuousDataset.__init__(self, config)
        StructureWeightDataset.__init__(self, config)


class ContinuousStateWeightDataset(ContinuousDataset):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        ContinuousDataset.__init__(self, config)
        self.max_num_points = config.max_num_points
        self.reduce = config.reduce
        self.compounded = config.compounded
        self.total = config.total
        if not self.total and not self.compounded:
            raise ValueError("[total] and [compounded] are mutually exclusive.")

    def _get_weights_(self):
        print('Enter _get_weights_() in continuous_dataset.py')#; pdb.set_trace()
        if self.total: #True
            if self.m_networks.is_weighted: #False
                weights = StrengthContinuousGlobalStateWeight(
                    reduce=self.reduce, bias=self.bias
                )
            else:
                weights = ContinuousGlobalStateWeight( #转到datasets/weights/continuous.py 
                    reduce=self.reduce, bias=self.bias
                ) # 这个很快，不耗时
        else:
            if self.m_networks.is_weighted and self.compounded:
                weights = StrengthContinuousCompoundStateWeight(
                    reduce=self.reduce, bias=self.bias
                )
            elif self.m_networks.is_weighted and not self.compounded:
                weights = StrengthContinuousStateWeight(
                    reduce=self.reduce, bias=self.bias
                )
            elif not self.m_networks.is_weighted and self.compounded:
                weights = ContinuousCompoundStateWeight(
                    reduce=self.reduce, bias=self.bias
                )
            else:
                weights = ContinuousStateWeight(bias=self.bias)
        print('Next line: weights.compute')
        weights.compute(self, verbose=self.verbose) #转到datasets/weights/weight.py的def compute
        return weights
