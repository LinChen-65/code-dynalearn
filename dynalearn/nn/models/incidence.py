import torch
import torch.nn as nn

from .gnn import GraphNeuralNetwork
from dynalearn.config import Config
from dynalearn.nn.loss import weighted_mse

import pdb

class IncidenceEpidemicsGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        print('Entered IncidenceEpidemicsGNN.__init__()')
        #pdb.set_trace() #remove_pdb_1227
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        if "is_weighted" in config.__dict__ and config.is_weighted:
            edgeattr_size = 1
            nodeattr_size = 1
        else:
            edgeattr_size = 0
            nodeattr_size = 0
        self.num_states = config.num_states
        GraphNeuralNetwork.__init__(
            self,
            1,
            1,
            lag=config.lag,
            #nodeattr_size=1,#original
            nodeattr_size=nodeattr_size, #20211226
            edgeattr_size=edgeattr_size,
            out_act="identity",
            normalize=True,
            config=config,
            **kwargs
        )
        print('Leave IncidenceEpidemicsGNN.__init__()') #回到dynamics/trainable/incidence.py(23)__init__()

    def loss(self, y_true, y_pred, weights): #从nn/models/model.py的def _do_batch_()的loss += self.loss(y_true, y_pred, w)跳过来
        return weighted_mse(y_true, y_pred, weights=weights)
