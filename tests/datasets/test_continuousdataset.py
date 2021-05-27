import networkx as nx
import numpy as np
import os
import torch
import unittest
import warnings

warnings.filterwarnings("ignore")

from templates import TemplateDatasetTest
from dynalearn.config import ExperimentConfig


class ContinuousDatasetTest(TemplateDatasetTest, unittest.TestCase):
    def get_config(self):
        return ExperimentConfig.test(config="continuous")

    def get_input_shape(self):
        return self.num_nodes, self.num_states, self.lag

    #     self.num_networks = 2
    #     self.num_samples = 10
    #     self.num_nodes = 10
    #     self.p = 10
    #     self.batch_size = 5
    #     self.config.train_details.num_networks = self.num_networks
    #     self.config.train_details.num_samples = self.num_samples
    #     self.config.networks.num_nodes = self.num_nodes
    #     self.config.networks.p = self.p
    #     self.exp = Experiment(self.config, verbose=0)
    #     self.dataset = self.exp.dataset
    #     self.dataset.setup(self.exp)
    #     data = self.dataset._generate_data_(self.exp.train_details)
    #     self.dataset.data = data
    #
    # def test_get_indices(self):
    #     indices = self.dataset.indices
    #     ref_indices = list(range(self.num_networks * self.num_samples))
    #     self.assertEqual(self.num_networks * self.num_samples, len(indices))
    #     self.assertEqual(ref_indices, list(indices.keys()))
    #
    # def test_get_weights(self):
    #     weights = self.dataset.weights
    #     for i in range(self.num_networks):
    #         self.assertEqual(weights[i].data.shape, (self.num_samples, self.num_nodes))
    #
    # def test_partition(self):
    #     dataset = self.dataset.partition(0.5)
    #     for i in range(self.num_networks):
    #         np.testing.assert_array_equal(self.dataset.networks[i], dataset.networks[i])
    #         np.testing.assert_array_equal(
    #             self.dataset.inputs[i].data, dataset.inputs[i].data
    #         )
    #         np.testing.assert_array_equal(
    #             self.dataset.targets[i].data, dataset.targets[i].data
    #         )
    #         index1 = self.dataset.weights[i].data == 0.0
    #         index2 = dataset.weights[i].data > 0.0
    #         np.testing.assert_array_equal(index1, index2)
    #     return
    #
    # def test_next(self):
    #     it = iter(self.dataset)
    #     data = next(it)
    #     i = 0
    #     for data in self.dataset:
    #         i += 1
    #     self.assertEqual(self.num_samples * self.num_networks, i)
    #     (x, g), y, w = data
    #     x_ref = np.zeros(
    #         (self.num_nodes, self.dataset.num_states, self.dataset.window_size)
    #     )
    #     y_ref = np.zeros((self.num_nodes, self.dataset.num_states))
    #     y_ref[:, 0] = 1
    #     w_ref = np.ones(self.num_nodes) / self.num_nodes
    #     np.testing.assert_array_almost_equal(x_ref.shape, x.shape)
    #     np.testing.assert_array_almost_equal(y_ref.shape, y.shape)
    #     np.testing.assert_array_almost_equal(w_ref.shape, w.shape)
    #
    # def test_batch(self):
    #     batches = self.dataset.to_batch(self.batch_size)
    #     i = 0
    #     for b in batches:
    #         j = 0
    #         for bb in b:
    #             (x, g), y, w = bb
    #             self.assertEqual(type(x), torch.Tensor)
    #             self.assertEqual(type(g), nx.Graph)
    #             self.assertEqual(type(y), torch.Tensor)
    #             self.assertEqual(type(w), torch.Tensor)
    #             j += 1
    #             i += 1
    #             pass
    #         self.assertEqual(self.batch_size, j)
    #     self.assertEqual(self.num_samples * self.num_networks, i)


if __name__ == "__main__":
    unittest.main()