import networkx as nx
import numpy as np

from dynalearn.experiments.metrics import Metrics
from dynalearn.util import Verbose
from dynalearn.dynamics.trainable import VARDynamics

import pdb
import time
import torch.nn as nn


class ForecastMetrics(Metrics):
    def __init__(self, config):
        Metrics.__init__(self, config)
        #pdb.set_trace()
        self.num_steps = config.forecast.get("num_steps", [1])
        if isinstance(self.num_steps, int):
            self.num_steps = [self.num_steps]
        elif not isinstance(self.num_steps, list):
            self.num_steps = list(self.num_steps)

    def get_model(self, experiment):
        raise NotImplementedError()

    def initialize(self, experiment):
        return

    def compute(self, experiment, verbose=Verbose()): #从experiments/experiment.py的compute_metrics()跳过来
        print('Entered compute() in experiments/metrics/forecast.py.')#; pdb.set_trace()
        self.verbose = verbose
        self.initialize(experiment)

        self.model = self.get_model(experiment)
        datasets = {
            "total": experiment.dataset,
            "train": experiment.dataset,
            "val": experiment.val_dataset,
            "test": experiment.test_dataset,
        }
        datasets = {
            k: self._get_data_(v, total=k == "total")
            for k, v in datasets.items()
            if v is not None
        }
        nobs = [v.shape[0] for k, v in datasets.items()] #各数据集的time_step数量：[59, 50, 50, 9]
        self.num_updates = np.sum(
            [[s * (n - s + 1) for s in self.num_steps] for n in nobs] #self.num_steps=1, self.num_updates=1*(59+50+50+9)=168
        )
        pb = self.verbose.progress_bar(self.__class__.__name__, self.num_updates)
        for k, v in datasets.items():
            total = k == "total"
            for s in self.num_steps: #self.num_steps=[1]
                start = time.time()
                self.data[f"{k}-{s}"] = self._get_forecast_(v, s, pb) ###这里有bug: *** ValueError: could not broadcast input array from shape (32,31656) into shape (1,31656)#f"{k}-{s}"形如"total-1""test-1" #(20220110)这里forecast后最后一天所有CBG的值都是0#已解决
                print(k,', Forecast takes time: ' ,time.time()-start)

        if pb is not None:
            pb.close()

        print('Leave compute() in experiments/metrics/forecast.py.')#; pdb.set_trace()
        self.exit(experiment)

    def _get_forecast_(self, dataset, num_steps=1, pb=None): #self: experiment.metrics['GNNForecastMetrics']
        print('Entered _get_forecast_() in experiments/metrics/forecast.py.');pdb.set_trace(); 
        if dataset.shape[0] - num_steps + 1 < 0:
            return np.zeros((0, *dataset.shape[1:-1]))
        y = np.zeros((dataset.shape[0] - num_steps + 1, *dataset.shape[1:-1])) # y.shape:(59, 31656, 1)

        #for i, x in enumerate(dataset[:-num_steps]): #original #有问题，当num_steps=1时，取出的dataset少了最后一个time step，这就导致最后一个time step没有被预测，所有值都是0
        if(num_steps==1):
            for i, x in enumerate(dataset):
                #if(i==0):continue
                for t in range(num_steps): #num_steps=1 #original #20220115暂时注释
                #for t in range(1): #20220115 test
                    print('Will call self.model.sample(x).')
                    yy = self.model.sample(x) #调用dynamics/deterministic_epidemics/base.py(101)sample()
                    print('Done calling self.model.sample(x).')
                    x = np.roll(x, -1, axis=-1)
                    print('x.shape: ', x.shape, 'yy.shape: ', yy.shape) #x.shape:  (31656, 1, 5) yy.shape:  (31656, 32)
                    x.T[-1] = yy.T #original #20220115暂时注释
                    if pb is not None:
                        pb.update()
                y[i] = yy
        else:
            for i, x in enumerate(dataset[:,-num_steps+1]):
                for t in range(num_steps): #num_steps=1
                    yy = self.model.sample(x) #调用dynamics/deterministic_epidemics/base.py(101)sample()
                    x = np.roll(x, -1, axis=-1)
                    x.T[-1] = yy.T
                    if pb is not None:
                        pb.update()
                y[i] = yy
        print('Leave _get_forecast_() in experiments/metrics/forecast.py.'); pdb.set_trace()
        return y

    def _get_data_(self, dataset, total=False):
        if dataset is None:
            return
        data = dataset.inputs[0].data
        if not total:
            w = dataset.state_weights[0].data
            data = dataset.inputs[0].data[w > 0]
        return data


class GNNForecastMetrics(ForecastMetrics):
    def get_model(self, experiment):
        print('Entered GNNForecastMetrics.get_model() in experiments/metrics/forcast.py.')
        model = experiment.model
        model.network = experiment.dataset.networks[0].data
        print('Leave GNNForecastMetrics.get_model() in experiments/metrics/forcast.py.')
        return model

    



class TrueForecastMetrics(ForecastMetrics):#用网络底层的dynamics生成results
    def get_model(self, experiment):
        print('Entered TrueForecastMetrics.get_model() in experiments/metrics/forcast.py.')
        model = experiment.dynamics
        model.network = experiment.dataset.networks[0].data
        print('Leave TrueForecastMetrics.get_model() in experiments/metrics/forcast.py.')
        return model


class VARForecastMetrics(ForecastMetrics):
    def get_model(self, experiment):
        print('Entered VARForecastMetrics.get_model() in experiments/metrics/forcast.py.')
        pdb.set_trace()
        model = VARDynamics(experiment.model.num_states, lag=experiment.model.lag)
        model.network = experiment.dataset.networks[0].data
        c = experiment.dataset.state_weights[0].data > 0
        X = experiment.dataset.inputs[0].data[c]
        Y = experiment.dataset.targets[0].data[c]
        model.fit(X, Y=Y)
        print('Leave VARForecastMetrics.get_model() in experiments/metrics/forcast.py.')
        return model



class GNNEmbeddingMetrics(ForecastMetrics): #20220115
    def get_model(self, experiment):
        print('Entered GNNEmbeddingMetrics.get_model() in experiments/metrics/forcast.py.')
        model = experiment.model
        replacement = nn.Sequential()
        model.nn.out_layers = replacement
        '''
        model.nn.out_layers[0].layers[5] = Identity
        model.nn.out_layers[0].layers[4] = Identity
        model.nn.out_layers[0].layers[3] = Identity
        model.nn.out_layers[0].layers[2] = Identity
        model.nn.out_layers[0].layers[1] = Identity
        #model.nn.out_layers[0] = replacement
        '''
        #model =  nn.Sequential(*list(experiment.model.nn.children())[:3], experiment.model.nn.out_layers[0].layers[0])
        
        pdb.set_trace()
        model.network = experiment.dataset.networks[0].data
        print('Leave GNNEmbeddingMetrics.get_model() in experiments/metrics/forcast.py.')
        return model

    def compute(self, experiment, verbose=Verbose()): #从experiments/experiment.py的compute_metrics()跳过来
        print('Entered compute() in experiments/metrics/forecast.py.')#; pdb.set_trace()
        self.verbose = verbose
        self.initialize(experiment)

        self.model = self.get_model(experiment)
        datasets = {
            "total": experiment.dataset,
        }
        datasets = {
            k: self._get_data_(v, total=k == "total")
            for k, v in datasets.items()
            if v is not None
        }
        nobs = [v.shape[0] for k, v in datasets.items()] #各数据集的time_step数量：[59, 50, 50, 9]
        self.num_updates = np.sum(
            [[s * (n - s + 1) for s in self.num_steps] for n in nobs] #self.num_steps=1, self.num_updates=1*(59+50+50+9)=168
        )
        pb = self.verbose.progress_bar(self.__class__.__name__, self.num_updates)
        for k, v in datasets.items():
            total = k == "total"
            for s in self.num_steps: #self.num_steps=[1]
                start = time.time()
                self.data[f"{k}-{s}"] = self._get_forecast_(v, s, pb) ###这里有bug: *** ValueError: could not broadcast input array from shape (32,31656) into shape (1,31656)#f"{k}-{s}"形如"total-1""test-1" #(20220110)这里forecast后最后一天所有CBG的值都是0#已解决
                print(k,', Forecast takes time: ' ,time.time()-start)

        if pb is not None:
            pb.close()

        print('Leave compute() in experiments/metrics/forecast.py.')#; pdb.set_trace()
        self.exit(experiment)

    def _get_forecast_(self, dataset, num_steps=1, pb=None): #self: experiment.metrics['GNNForecastMetrics']
        print('Entered _get_forecast_() in experiments/metrics/forecast.py.');pdb.set_trace(); 
        if dataset.shape[0] - num_steps + 1 < 0:
            return np.zeros((0, *dataset.shape[1:-1]))
        y = np.zeros((dataset.shape[0] - num_steps + 1, *dataset.shape[1:-1])) # y.shape:(59, 31656, 1)

        #for i, x in enumerate(dataset[:-num_steps]): #original #有问题，当num_steps=1时，取出的dataset少了最后一个time step，这就导致最后一个time step没有被预测，所有值都是0
        if(num_steps==1):
            for i, x in enumerate(dataset):
                #if(i==0):continue
                #for t in range(num_steps): #num_steps=1 #original #20220115暂时注释
                for t in range(1): #20220115 test
                    print('Will call self.model.sample(x).')
                    yy = self.model.sample(x) #调用dynamics/deterministic_epidemics/base.py(101)sample()
                    print('Done calling self.model.sample(x).')
                    x = np.roll(x, -1, axis=-1)
                    print('x.shape: ', x.shape, 'yy.shape: ', yy.shape) #x.shape:  (31656, 1, 5) yy.shape:  (31656, 32)
                    #x.T[-1] = yy.T #original #20220115暂时注释
                    if pb is not None:
                        pb.update()
                #y[i] = yy
        else:
            for i, x in enumerate(dataset[:,-num_steps+1]):
                for t in range(num_steps): #num_steps=1
                    yy = self.model.sample(x) #调用dynamics/deterministic_epidemics/base.py(101)sample()
                    x = np.roll(x, -1, axis=-1)
                    x.T[-1] = yy.T
                    if pb is not None:
                        pb.update()
                y[i] = yy
        print('Leave _get_forecast_() in experiments/metrics/forecast.py.'); pdb.set_trace()
        #return y
        return yy