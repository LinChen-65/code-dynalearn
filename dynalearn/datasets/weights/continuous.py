import networkx as nx
import numpy as np

from scipy.stats import gmean
from .weight import Weight
from .kde import KernelDensityEstimator

import pdb
import time

NUM_CBGS = 2943 #temporary; 20 for Spanish, 2943 for SanFrancisco

class ContinuousStateWeight(Weight): #继承了Weight类的_add_features_()方法
    def __init__(self, name="weights", reduce=True, bias=1.0):
        self.reduce = reduce
        Weight.__init__(self, name=name, max_num_samples=10000, bias=bias)

    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [dataset.inputs[i].data.shape[0] for i in range(dataset.networks.size)]
        )

    def _reduce_node_state_(self, index, states, network):
        print('ContinuousStateWeight._reduce_node_state_() in datasets/weights/continuous.py.')
        x = states[index].reshape(-1)
        if self.reduce:
            x = np.array([x.sum()])
        return x

    def _reduce_total_state_(self, states, network):
        return

    def _reduce_node_(self, index, network):
        return

    def _reduce_network_(self, network):
        print('ContinuousStateWeight._reduce_network_() in datasets/weights/continuous.py. Does nothing!')
        return

    def _get_features_(self, network, states, pb=None): #self: StrengthContinuousGlobalStateWeight
        print('Entered _get_features_ in datasets/weights/continuous.py')
        
        x = self._reduce_network_(network) #继承ContinuousStateWeight._reduce_network_() #does nothing
        if x is not None:
            self._add_features_("network", x)
        start = time.time()
        for i in network.nodes: #30000个节点，用时约372s
            #if(i%5000==0): print(i)
            k = network.degree(i)
            self._add_features_(("degree", int(k)))
            x = self._reduce_node_(i, network) #StrengthContinuousGlobalStateWeight._reduce_node_()
            if x is not None:
                self._add_features_(("node", int(k)), x)
        print('Time used: ', time.time()-start)
        #pdb.set_trace()

        start = time.time()
        num_cbgs = NUM_CBGS #20 #2943 #test
        sub_states = states[:,:num_cbgs,:,:]
        #for i, s in enumerate(states): #s:即x，即一段历史infection(5天) #i: num_days (Spanish:446) #states.shape:(446, 52, 1, 5)
        for i, s in enumerate(sub_states): #s:即x，即一段历史infection(5天) #i: num_days (Spanish:446) #states.shape:(446, 52, 1, 5)
            if(i%10==0):print('***',i,'***')
            y = self._reduce_total_state_(s, network) #return states.sum(0).reshape(-1) 
            if y is not None:
                self._add_features_("total_state", y) #所有节点总感染
            for j, ss in enumerate(s): #len([j for j, ss in enumerate(s)]) = num_nodes #s.shape:(31656, 1, 5)
                # 这个loop非常耗时！！！
                #if(j%5000==0):print(j)
                k = network.degree(j)
                x = self._reduce_node_state_(j, s, network) #does nothing (而且self.reduce==False)
                if x is not None:
                    self._add_features_(("node_state", int(k)), x) #(key,value) #每个节点感染

        '''
        #original
        for i, s in enumerate(states): #s:即x，即一段历史infection(5天) #i: num_days (Spanish:446)
            y = self._reduce_total_state_(s, network) #return states.sum(0).reshape(-1) 
            if y is not None:
                self._add_features_("total_state", y) #所有节点总感染
            for j, ss in enumerate(s): #len([j for j, ss in enumerate(s)]) = num_nodes #s.shape:(31656, 1, 5)
                k = network.degree(j)
                x = self._reduce_node_state_(j, s, network) #does nothing (而且self.reduce==False)
                if x is not None:
                    self._add_features_(("node_state", int(k)), x) #(key,value) #每个节点感染
            if pb is not None:
                pb.update()
        '''
        print('Time used: ', time.time()-start)
        print('Leave _get_features_ in datasets/weights/continuous.py')
        #pdb.set_trace()

    def _get_weights_(self, network, states, pb=None): #应该是根据sampled node的feature稀有程度，计算importance weight (所以用了核密度估计) #type(self):<class 'dynalearn.datasets.weights.continuous.StrengthContinuousGlobalStateWeight'>
        #weights = np.ones((states.shape[0], states.shape[1])) #sampler里有归一化，故这里不用归一化 #test
        num_nodes = states.shape[1]
        num_cbgs = NUM_CBGS #20 #2943 #test
        print('num_nodes: ', num_nodes, ', num_cbgs: ', num_cbgs)
        #weights = np.tile(np.concatenate((np.ones(num_cbgs), np.zeros(num_nodes-num_cbgs)), axis=0), (states.shape[0],1)) #为了加速的权宜之计，把所有CBG的weight置为1，所有POI的weight置为0，potential harm是cases最多的CBG测试误差大
        #pdb.set_trace()

        #20220111
        weights = np.zeros((states.shape[0], states.shape[1])) #(num_days, num_nodes) #states.shape:(59, 31656, 1, 5)
        z = 0
        kde = {}
        pp = {}
        for k, v in self.features.items():
            if k[0] == "degree":
                z += v
            else:
                kde[k] = KernelDensityEstimator(
                    samples=v, max_num_samples=self.max_num_samples
                )
        g_feats = self._reduce_network_(network) #does nothing
        if g_feats is not None:
            p_g = kde["network"].pdf(g_feats)
        else:
            p_g = 1.0
        for i, s in enumerate(states): #[i for i, s in enumerate(states)]输出[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
            s_feats = self._reduce_total_state_(s, network) #s.shape:(31656, 1, 5) #s_feats.shape: (5,) #return states.sum(0).reshape(-1) 
            if s_feats is not None:
                p_s = kde["total_state"].pdf(s_feats)
            else:
                p_s = 1.0
            #print('In continuous.py'); #pdb.set_trace()
            #for j, ss in enumerate(s): #original, 会遍历所有31656个节点(包括CBG和POI)
            for j, ss in enumerate(s[:num_cbgs,:,:]): #20220111 #ss.shape:(1,5)
                k = network.degree(j)
                p_k = self.features[("degree", k)] / z

                #ss_feats = self._reduce_node_state_(j, s, network) #original, 我怀疑这里写错了，s应为ss，但既然这个函数返回None似乎也没影响 #s没错 #_reduce_node_state_() does nothing.
                ss_feats = self._reduce_node_state_(j, s, network) #(20220111)修改StrengthContinuousGlobalStateWeight的_reduce_node_state_()
                if ss_feats is not None:
                    p_ss = gmean(kde[("node_state", k)].pdf(ss_feats))
                else:
                    p_ss = 1.0

                n_feats = self._reduce_node_(j, network)
                if n_feats is not None:
                    p_n = kde[("node", k)].pdf(n_feats)
                else:
                    p_n = 1.0

                weights[i, j] = p_k * p_s * p_ss * p_n * p_g
            
        #pdb.set_trace()
        '''
        #original
        weights = np.zeros((states.shape[0], states.shape[1]))
        z = 0
        kde = {}
        pp = {}
        for k, v in self.features.items():
            if k[0] == "degree":
                z += v
            else:
                kde[k] = KernelDensityEstimator(
                    samples=v, max_num_samples=self.max_num_samples
                )
        g_feats = self._reduce_network_(network)
        if g_feats is not None:
            p_g = kde["network"].pdf(g_feats)
        else:
            p_g = 1.0
        for i, s in enumerate(states):
            s_feats = self._reduce_total_state_(s, network)
            s_feats = self._reduce_total_state_(s, network)
            if s_feats is not None:
                p_s = kde["total_state"].pdf(s_feats)
            else:
                p_s = 1.0
            #print('In continuous.py'); #pdb.set_trace()
            for j, ss in enumerate(s):
                k = network.degree(j)
                p_k = self.features[("degree", k)] / z

                ss_feats = self._reduce_node_state_(j, s, network)
                if ss_feats is not None:
                    p_ss = gmean(kde[("node_state", k)].pdf(ss_feats))
                else:
                    p_ss = 1.0

                n_feats = self._reduce_node_(j, network)
                if n_feats is not None:
                    p_n = kde[("node", k)].pdf(n_feats)
                else:
                    p_n = 1.0

                weights[i, j] = p_k * p_s * p_ss * p_n * p_g
            #if pb is not None:
            #    pb.update()
        '''
        return weights #回到datasets/weights/weight.py的compute_weights()


class ContinuousGlobalStateWeight(ContinuousStateWeight): #从datasets/continuous_dataset.py跳过来
    def _reduce_node_state_(self, index, states, network):
        print('ContinuousGlobalStateWeight._reduce_node_state_() in datasets/weights/continuous.py.')
        return

    def _reduce_total_state_(self, states, network):
        return states.sum(0).reshape(-1)


class StrengthContinuousGlobalStateWeight(ContinuousStateWeight): #run-covid-mine.py会进入这里
    def _reduce_node_state_(self, index, states, network):
        '''
        #original
        print('StrengthContinuousGlobalStateWeight._reduce_node_state_() in datasets/weights/continuous.py.')
        return
        '''
        #(20220111)沿用ContinuousStateWeight的同名函数
        x = states[index].reshape(-1)
        if self.reduce:
            x = np.array([x.sum()])
        return x
        

    def _reduce_total_state_(self, states, network):
        return states.sum(0).reshape(-1)

    def _reduce_node_(self, index, network):
        #print('Entered _reduce_node_() in datasets/weights/continuous.py.')
        s = np.array([0.0])
        for l in network.neighbors(index):
            if "weight" in network.data.edges[index, l]:
                s += network.data.edges[index, l]["weight"]
            else:
                s += np.array([1.0])
        #print('Leave _reduce_node_() in datasets/weights/continuous.py.')
        return s.reshape(-1)


class StrengthContinuousStateWeight(ContinuousStateWeight):
    def _reduce_node_state_(self, index, states, network):
        x = states[index].reshape(-1)
        if self.reduce:
            x = np.array([x.sum()])
        s = np.array([0.0])
        for l in network.neighbors(index):
            if "weight" in network.data.edges[index, l]:
                s += network.data.edges[index, l]["weight"]
            else:
                s += np.array([1.0])
        return np.concatenate([x, s])


class ContinuousCompoundStateWeight(ContinuousStateWeight):
    def _reduce_node_state_(self, index, states, network):
        x = []
        _x = states[index].reshape(-1)
        if self.reduce:
            _x = np.array([_x.sum()])
        for l in network.neighbors(index):
            _y = states[l].reshape(-1)
            if self.reduce:
                _y = np.array([_y.sum()])
            x.append(np.concatenate([_x, _y]))
        return x


class StrengthContinuousCompoundStateWeight(ContinuousStateWeight):
    def _reduce_node_state_(self, index, states, network):
        x = []
        s = states[index]
        for l in network.neighbors(index):
            _x = s.reshape(-1)
            _y = states[l].reshape(-1)
            if "weight" in network.data.edges[index, l]:
                _w = np.array([network.data.edges[index, l]["weight"]])
            else:
                _w = np.array([1.0])
            if self.reduce:
                _x = np.array([_x.sum()])
                _y = np.array([_y.sum()])
            x.append(np.concatenate([_x, _y, _w]))
        return x
