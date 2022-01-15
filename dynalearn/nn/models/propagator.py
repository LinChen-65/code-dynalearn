import numpy as np
import torch

from torch_geometric.nn.conv import MessagePassing
from dynalearn.util import onehot

import pdb

class Propagator(MessagePassing):
    def __init__(self, num_states=None):
        MessagePassing.__init__(self, aggr="add")
        self.num_states = num_states

    def forward(self, x, edge_index, w=None):
        print('Entered forward() in nn/models/propagator.py.') #20220115
        pdb.set_trace()

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.LongTensor(edge_index)
        if isinstance(w, np.ndarray):
            assert w.shape[0] == edge_index.shape[-1]
            w = torch.Tensor(w).view(-1, 1)
        if isinstance(self.num_states, int):
            x = onehot(x, num_class=self.num_states)
        else:
            x = x.view(-1, 1)
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()
            if w is not None:
                w = w.cuda()
        print('Leave forward() in nn/models/propagator.py.') #20220115
        return self.propagate(edge_index, x=x, w=w).T

    def message(self, x_j, w=None):
        if w is None:
            return x_j
        else:
            out = w * x_j
            return out

    def update(self, x):
        return x
