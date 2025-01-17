import networkx as nx
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import tqdm
import psutil

from abc import abstractmethod
from dynalearn.config import Config
from dynalearn.nn.callbacks import CallbackList
from dynalearn.nn.history import History
from dynalearn.nn.optimizers import get as get_optimizer
from dynalearn.util import Verbose, LoggerDict

import pdb

NUM_CBGS = 2943 #temporary; 20 for Spanish, 2943 for SanFrancisco

class Model(nn.Module):
    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = config or Config(**kwargs)
        self.get_optimizer = get_optimizer(config.optimizer)
        self.history = History()
        if "network_layers" not in self.config.__dict__:
            self.config.network_layers = None

    @abstractmethod
    def forward(self, x, network_attr):
        raise NotImplemented()

    @abstractmethod
    def loss(self, y_true, y_pred, weights):
        raise NotImplemented()

    def fit(
        self,
        dataset,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        val_dataset=None,
        metrics={},
        callbacks=None,
        loggers=None,
        verbose=Verbose(),
        retrieve_embedding=False, #20220115
    ):
        print('Entered fit() in nn/models/model.py.')
        pdb.set_trace()
        self.train() #Set the module in training mode
        callbacks = callbacks or CallbackList()
        if isinstance(callbacks, list):
            callbacks = CallbackList(callbacks)
        for c in callbacks:
            if "verbose" in c.__dict__:
                c.verbose = verbose

        loggers = loggers or LoggerDict()

        callbacks.set_params(self)
        callbacks.set_model(self)
        callbacks.on_train_begin()

        self.transformers.setUp(dataset)
        for i in range(epochs):
            callbacks.on_epoch_begin(self.history.epoch)
            t0 = time.time()
            #pdb.set_trace()
            self._do_epoch_(
                dataset, batch_size=batch_size, callbacks=callbacks, verbose=verbose, retrieve_embedding=retrieve_embedding #20220115
            )

            train_metrics = self.evaluate(dataset, metrics=metrics, verbose=verbose) #这里有bug: TypeError: 'NoneType' object is not subscriptable #已解决
            if val_dataset is not None:
                val_metrics = self.evaluate(  #这里有bug: TypeError: 'NoneType' object is not subscriptable, 说明是val_dataset的问题
                    val_dataset, metrics=metrics, name="val", verbose=verbose
                )
            else:
                val_metrics = {}

            t1 = time.time()
            loggers.on_task_update("training")
            logs = {"epoch": self.history.epoch + 1, "time": t1 - t0}
            logs.update(train_metrics)
            logs.update(val_metrics)
            self.history.update_epoch(logs)
            callbacks.on_epoch_end(self.history.epoch, logs)
            verbose(self.history.display())

        callbacks.on_train_end(self.history._epoch_logs)
        self.eval() #Set the module in evaluation mode

    def _do_epoch_(
        self, dataset, batch_size=1, callbacks=CallbackList(), verbose=Verbose(),retrieve_embedding=False, #20220115
    ):
        #print('Entered _do_epoch_() in nn/models/model.py.')
        #pdb.set_trace()
        epoch = self.history.epoch
        num_updates = len(dataset) // batch_size
        if len(dataset) % batch_size > 0:
            num_updates += 1
        pb = verbose.progress_bar("Epoch %d" % (epoch), num_updates)
        
  
        self.train() #Set the module in training mode
        for batch in dataset.to_batch(batch_size): #调用datasets/dataset.py的to_batch()->调用datasets/sampler.py
            #pdb.set_trace()
            self.optimizer.zero_grad()

            callbacks.on_batch_begin(self.history.batch)
            t0 = time.time()
            loss = self._do_batch_(batch,retrieve_embedding=retrieve_embedding) #20220115
            t1 = time.time()
            logs = {
                "batch": self.history.batch + 1,
                "loss": loss.cpu().detach().numpy(),
                "time": t1 - t0,
            }
            self.history.update_batch(logs)
            loss.backward()
            callbacks.on_backward_end(self.history.batch)

            self.optimizer.step() #20220103 #这一句后出现nan #部分p.grad=NaN #已解决
            callbacks.on_batch_end(self.history.batch, logs)
            if pb is not None:
                pb.set_description(f"Epoch {epoch} loss: {loss:.4f}")
                pb.update()

        if pb is not None:
            pb.set_description(f"Epoch {epoch}")
            pb.close()
        self.eval()

    def _do_batch_(self, batch, retrieve_embedding=False): #20220115 #type(self):<class 'dynalearn.nn.models.incidence.IncidenceEpidemicsGNN'>
        #print('Entered _do_batch_() in nn/models/model.py.')
        #pdb.set_trace()
        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss = loss.cuda()
        num_samples = 0
        for data in batch:
            #pdb.set_trace()#20220111
            y_true, y_pred, w = self.prepare_output(data,retrieve_embedding=retrieve_embedding) #用normalizer归一化，且用CUDAtransformer转为cuda()
            #loss += self.loss(y_true, y_pred, w) #original
            num_cbgs = NUM_CBGS #2943 #20 #test
            #loss += self.loss(y_true[:num_cbgs], y_pred[:num_cbgs], w[:num_cbgs]) #mask: only compute cbg losses #method1, fail #20220103
            y_true[num_cbgs:] = 0 #method2 #20220103 
            y_pred[num_cbgs:] = 0
            loss += self.loss(y_true, y_pred, w) #转到nn/models/incidence.py
            num_samples += 1
        return loss / num_samples

    def evaluate(self, dataset, metrics={}, name=None, verbose=Verbose()):
        #pdb.set_trace()
        if name is not None:
            prefix = name + "_"
        else:
            name = "train"
            prefix = ""
        metrics["loss"] = self.loss

        logs = {}
        for m in metrics:
            logs[prefix + m] = 0

        self.eval()

        pb = verbose.progress_bar(f"Evaluating {name}", len(dataset) + 1)
        #pdb.set_trace() 

        norm = 0.0
        #verbose_count = 0 #(20220107)只打印10次运行时间
        #batch_count = 0 #20220109
        for data in dataset:
            break
        for data in dataset: #这里有bug: TypeError: 'NoneType' object is not subscriptable #(20220108)已解决
            #batch_count += 1
            if('data' not in locals()):
                continue
            start_time = time.time()
            y_true, y_pred, w = self.prepare_output(data) #这是怎么从data里把w拿出来的？y_true和y_pred形状均为(num_nodes,1)(31656,1)
            #if(verbose_count<10): #20220107
            #    print('Forwarding this batch uses time: ', time.time()-start_time); 
            #    verbose_count += 1
            z = w.sum().cpu().detach().numpy()
            norm += z
            for m in metrics:
                logs[prefix + m] += (
                    z * metrics[m](y_true, y_pred, w).cpu().detach().numpy()
                )
            if pb is not None:
                pb.update()
        #print('batch_count: ', batch_count);pdb.set_trace()
        if pb is not None:
            pb.close()

        for m in metrics:
            logs[prefix + m] = logs[prefix + m] / norm
        return logs

    def prepare_output(self, data,retrieve_embedding=False): #一个时间步
        #pdb.set_trace()
        data = self.transformers.forward(data) #这是怎么从data里把w拿出来的？ #type(self.transformers):<class 'dynalearn.nn.transformers.batch.BatchNormalizer'>, 跳到nn/transformers/batch.py的class BatchNormalizer的def forward()
        (x, g), y, w = data
        y_true = y
        #if(retrieve_embedding):
        #    print('Stop here!')
        #    pdb.set_trace()
        y_pred = self.forward(x, g) #这里有bug: RuntimeError: mat1 dim 1 must match mat2 dim 0 #已解决 #这里的forward调用gnn.py的def forward(self, x, network_attr)
        
        return y_true, y_pred, w

    def get_weights(self):
        return self.state_dict()

    def save_weights(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load_weights(self, path):
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict) #nn原生
        if torch.cuda.is_available():
            self.cuda()
            self.transformers = self.transformers.cuda()

    def save_optimizer(self, path):
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, path)

    def load_optimizer(self, path):
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        state_dict = torch.load(path, map_location=device)
        self.optimizer.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self = self.cuda()
            self.transformers = self.transformers.cuda()

    def save_history(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    def load_history(self, path):
        with open(path, "rb") as f:
            self.history = pickle.load(f)

    def num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += torch.tensor(p.size()).prod()
        return num_params

    def grad_norm(self):
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm
