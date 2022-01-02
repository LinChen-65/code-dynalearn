import torch

from .transformer import TransformerDict, CUDATransformer
from .normalizer import InputNormalizer, TargetNormalizer, NetworkNormalizer
from dynalearn.util import get_node_attr

import pdb

class BatchNormalizer(TransformerDict):
    def __init__(
        self,
        input_size=0,
        target_size=0,
        node_size=0,
        edge_size=0,
        layers=None,
        auto_cuda=True,
    ):
        transformer_dict = {"t_cuda": CUDATransformer()}
        if input_size is not None:
            transformer_dict["t_inputs"] = InputNormalizer(
                input_size, auto_cuda=auto_cuda
            )
        else:
            transformer_dict["t_inputs"] = CUDATransformer()

        if target_size is not None:
            transformer_dict["t_targets"] = TargetNormalizer(target_size)
        else:
            transformer_dict["t_targets"] = CUDATransformer()

        transformer_dict["t_networks"] = NetworkNormalizer( #转到nn/transformers/normalizer.py
            node_size, edge_size, layers=layers, auto_cuda=auto_cuda
        )

        TransformerDict.__init__(self, transformer_dict)

    def forward(self, data):
        #pdb.set_trace() #remove_pdb_1227
        (x, g), y, w = data
        x = self["t_inputs"].forward(x) #这里的forward应该是torch原生函数
        g = self["t_networks"].forward(g) #这里的forward转到nn/transformers/normalizer.py的def forward(self, g) #g变成一个tuple，三个元素分别是edge_index, edge_attr, node_attr
        y = self["t_targets"].forward(y)
        w = self["t_cuda"].forward(w)
        return (x, g), y, w #回到nn/models/model.py(174)prepare_output()

    def backward(self, data):
        (x, g), y, w = data
        x = self["t_inputs"].backward(x)
        g = self["t_networks"].backward(g)
        y = self["t_targets"].backward(y)
        w = self["t_cuda"].backward(w)
        return (x, g), y, w
