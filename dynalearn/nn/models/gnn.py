import numpy as np
import torch
import torch.nn as nn

from torch.nn import Parameter, Sequential, Linear, Identity
from .dgat import DynamicsGATConv
from .model import Model
from .util import (
    get_in_layers,
    get_out_layers,
    reset_layer,
    ParallelLayer,
    LinearLayers,
)
from torch.nn.init import kaiming_normal_
from dynalearn.nn.activation import get as get_activation
from dynalearn.nn.models.getter import get as get_gnn_layer
from dynalearn.nn.transformers import BatchNormalizer
from dynalearn.config import Config

import pdb

class GraphNeuralNetwork(Model):
    def __init__(
        self,
        in_size,
        out_size,
        lag=1,
        nodeattr_size=0,
        edgeattr_size=0,
        out_act="identity",
        normalize=False,
        config=None,
        **kwargs
    ):
        print('Entered GraphNeuralNetwork.__init__()')
        #pdb.set_trace() #remove_pdb_1227
        Model.__init__(self, config=config, **kwargs)

        self.in_size = self.config.in_size = in_size
        self.out_size = self.config.out_size = out_size
        self.out_act = self.config.out_act = out_act
        self.lag = self.config.lag = lag
        self.nodeattr_size = nodeattr_size
        self.edgeattr_size = edgeattr_size

        if self.nodeattr_size > 0 and "node_channels" in self.config.__dict__:
            self.node_layers = LinearLayers(
                [self.nodeattr_size] + self.config.node_channels,
                self.config.node_activation,
                self.config.bias,
            )
        else:
            self.node_layers = Sequential(Identity())
            self.config.node_channels = [self.nodeattr_size]

        if self.edgeattr_size > 0 and "edge_channels" in self.config.__dict__:
            template = lambda: LinearLayers(
                [self.edgeattr_size] + self.config.edge_channels,
                self.config.edge_activation,
                self.config.bias,
            )
            if self.config.is_multiplex:
                self.edge_layers = ParallelLayer(
                    template, keys=self.config.network_layers
                )
            else:
                self.edge_layers = template()
        else:
            self.edge_layers = Sequential(Identity())

        self.in_layers = get_in_layers(self.config)
        self.gnn_layer = get_gnn_layer(self.config) #self.gnn_layer = get_gnn_layer(self.config) <== from dynalearn.nn.models.getter import get as get_gnn_layer
        self.out_layers = get_out_layers(self.config)

        if normalize: #True
            input_size = in_size #1
            target_size = out_size #1
        else:
            input_size = 0
            target_size = 0
        self.transformers = BatchNormalizer( #转到nn/transformers/batch.py
            input_size=input_size,
            target_size=target_size,
            edge_size=edgeattr_size,
            node_size=nodeattr_size,
            layers=self.config.network_layers,
        )

        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()
        print('Leave GraphNeuralNetwork.__init__()') #回到/nn/models/incidence.py(37)__init__()
        #pdb.set_trace()

    def forward(self, x, network_attr):
        #pdb.set_trace() #remove_pdb_1227 #add_pdb_1228 #remove_pdb_1228
        edge_index, edge_attr, node_attr = network_attr
        x = self.in_layers(x) #in_layers是RNN layer，(1, 16) #x从[52,1]变成[52,16]
        node_attr = self.node_layers(node_attr) #node_attr从[52,1]变成[52,4]
        edge_attr = self.edge_layers(edge_attr) #edge_attr从[467,1]变成[467,8]

        #x = self.merge_nodeattr(x, node_attr) #original
        #x = x #20211227
        x = self.merge_nodeattr(x, node_attr) #20220101 #

        if self.config.gnn_name == "DynamicsGATConv":
            # nn/models/dgat.py/class DynamicsGATConv(MessagePassing)的def forward <== self.gnn_layer = get_gnn_layer(self.config) <== from dynalearn.nn.models.getter import get as get_gnn_layer
            x = self.gnn_layer(x, edge_index, edge_attr=edge_attr) #这里有bug, 20211227 #已解决, 20211227
        else:
            x = self.gnn_layer(x, edge_index)
        if isinstance(x, tuple):
            x = x[0]
        x = self.out_layers(x)
        return x

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_
        reset_layer(self.edge_layers, initialize_inplace=initialize_inplace)
        reset_layer(self.in_layers, initialize_inplace=initialize_inplace)
        reset_layer(self.out_layers, initialize_inplace=initialize_inplace)
        self.gnn_layer.reset_parameters()

    def merge_nodeattr(self, x, node_attr):
        if node_attr is None:
            return x
        # print(x.shape, node_attr.shape)
        assert x.shape[0] == node_attr.shape[0]
        n = x.shape[0]
        return torch.cat([x, node_attr.view(n, -1)], dim=-1)
