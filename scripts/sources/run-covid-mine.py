# python run-covid-mine.py gen_code
# python run-covid-mine.py 1
 
import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

import argparse
import dynalearn
import h5py
import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
import time
import tqdm
import pdb

from os.path import exists, join

from sklearn.metrics import mean_squared_error
from math import sqrt

###############################################################################
# Main variable settings

MSA_NAME = 'SanFrancisco' #test #20211223
NUM_CBGS = 2943 #SanFrancisco
#gen_code = 1

###############################################################################
# Functions

# return ground truth (targets)
def loading_prediction_targets(
    experiment, path_to_covid, gen_code=0, lag=1, lagstep=1, incidence=True, threshold=False
):
    if incidence:
        print('path_to_covid: ', path_to_covid)
        dataset = h5py.File(os.path.join(os.path.abspath('../..'),path_to_covid, "data_%s_gencode%s.h5"%(MSA_NAME,str(gen_code))), "r") #20211223
        num_states = 1
    else:
        dataset = h5py.File(os.path.join(path_to_covid, "spain-covid19.h5"), "r")
        num_states = 3

    Y = dataset['timeseries'][...]
    targets = np.zeros((Y.shape[0] - (lag - 1) * lagstep, Y.shape[1], num_states))
    for t in range(targets.shape[0]):
        y = Y[t + lag * lagstep - 1]
        if incidence:
            y = y.reshape(*y.shape, 1)
        targets[t] = y
    
    return targets

def loading_covid_data(
    experiment, path_to_covid, gen_code=0, lag=1, lagstep=1, incidence=True, threshold=False
):
    pre_exists = experiment.load_data(label_with_mode=False) #20220104, problem during training, 暂时不用 #20220108,debugged
    #experiment.dataset = experiment._dataset #20220104, 不对
    if(not pre_exists):
        print("Unable to load data, construct from scratch.")
        if incidence:
            #dataset = h5py.File(os.path.join(path_to_covid, "spain-covid19cases.h5"), "r")
            print('path_to_covid: ', path_to_covid)
            #dataset = h5py.File(os.path.join(os.path.abspath('../..'),path_to_covid, "spain-covid19-dataset.h5"), "r") #20211221
            #dataset = h5py.File(os.path.join(os.path.abspath('../..'),path_to_covid, "data_%s.h5"%(MSA_NAME)), "r") #20211223
            dataset = h5py.File(os.path.join(os.path.abspath('../..'),path_to_covid, "data_%s_gencode%s.h5"%(MSA_NAME,str(gen_code))), "r") #20211223
            num_states = 1
        else:
            dataset = h5py.File(os.path.join(path_to_covid, "spain-covid19.h5"), "r")
            num_states = 3

        X = dataset['timeseries'][...]
        Y = dataset['timeseries'][...]
        networks = dataset['networks']

        data = {
            "inputs": dynalearn.datasets.DataCollection(name="inputs"),
            "targets": dynalearn.datasets.DataCollection(name="targets"),
            "networks": dynalearn.datasets.DataCollection(name="networks"),
        }
        inputs = np.zeros((X.shape[0] - (lag - 1) * lagstep, X.shape[1], num_states, lag)) #X.shape[1]即num_nodes, X.shape[0]即num_timesteps
        targets = np.zeros((Y.shape[0] - (lag - 1) * lagstep, Y.shape[1], num_states))
        for t in range(inputs.shape[0]):
            x = X[t : t + lag * lagstep : lagstep]
            y = Y[t + lag * lagstep - 1]
            if incidence:
                x = x.reshape(*x.shape, 1)
                y = y.reshape(*y.shape, 1)
            x = np.transpose(x, (1, 2, 0))
            inputs[t] = x
            targets[t] = y
        #pdb.set_trace()

        data["inputs"].add(dynalearn.datasets.StateData(data=inputs))
        data["targets"].add(dynalearn.datasets.StateData(data=targets))
        start = time.time()
        data["networks"].add(dynalearn.datasets.NetworkData(data=networks)) #This line uses time: 2.80s
        #pop = data["networks"][0].data.node_attr["population"] #20211222注释掉,因为下文也没用到
        print('Initialize experiment.dataset.data with data.')
        experiment.dataset.data = data #This line uses time: 425.95s #experiment.dataset.data['inputs'].data_list[0].data.shape:(59, 52, 1, 5)
        print('This line uses time: ',time.time()-start); start = time.time()
        
        experiment.test_dataset = experiment.dataset.partition( #这个会改变weights
            type="cleancut", 
            #ti=335, #original
            ti=50,
            tf=-1
        ) #This line uses time: 0.002s
        experiment.partition_val_dataset() #This line uses time: 0.015s #这个会改变weights
        
    else:
        index = 0
        indices_dict = {}
        for i in range(experiment.dataset.data["networks"].size):
            for j in range(experiment.dataset.data["inputs"][i].size):
                indices_dict[index] = (i, j)
                index += 1
        experiment.dataset.indices = indices_dict
        experiment.test_dataset.indices = indices_dict
        experiment.val_dataset.indices = indices_dict 

    #return experiment #original
    #return experiment,targets #20211229
    return experiment, pre_exists #20220104

###############################################################################
## REQUIRED PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", type=str, metavar="NAME", help="Name of the experiment.", required=True
)
parser.add_argument(
    "--path_to_data",
    type=str,
    metavar="PATH",
    help="Path to the directory where to save the experiment.",
    required=True,
)
parser.add_argument(
    "--path_to_covid",
    type=str,
    metavar="PATH",
    help="Path to the directory where covid data is saved.",
    required=True,
)
parser.add_argument(
    "--path_to_best",
    type=str,
    metavar="PATH",
    help="Path to the model directory.",
    required=True,
)
parser.add_argument(
    "--path_to_summary",
    type=str,
    metavar="PATH",
    help="Path to the summaries directory.",
    required=True,
)
parser.add_argument(
    "--verbose",
    type=int,
    choices=[0, 1, 2],
    metavar="VERBOSE",
    help="Verbose.",
    default=0,
)

## OPTIONAL PARAMETERS
parser.add_argument(
    "--epochs",
    type=int,
    metavar="EPOCHS",
    help="Number of epochs to train.",
    default=30,
)
parser.add_argument(
    "--incidence",
    type=int,
    metavar="INCIDENCE",
    help="Using incidence to train the model.",
    default=1,
)
parser.add_argument(
    "--bias",
    type=float,
    metavar="BIAS",
    help="Exponent bias to use.",
    default=0.5,
)
parser.add_argument(
    "--val_fraction", #在figure-6/run-covid-mine.py设置了"val_fraction": 0.1
    type=float,
    metavar="VAL_FRACTION",
    help="Size of the validation dataset relative to the complete dataset.",
    default=0.01,
)
parser.add_argument(
    "--lag",
    type=int,
    metavar="lag",
    help="Size of the windows during training.",
    default=1,
)
parser.add_argument(
    "--lagstep",
    type=int,
    metavar="LAGSTEP",
    help="Step between windows during training.",
    default=1,
)
parser.add_argument(
    "--model",
    type=str,
    metavar="MODEL",
    help="GNN model to use.",
    default="DynamicsGATConv",
)
parser.add_argument(
    "--type",
    type=str,
    metavar="TYPE",
    help="Type of GNN model to use.",
    default="linear",
)
parser.add_argument(
    "--clean",
    type=int,
    metavar="CLEAN",
    help="Clean the experiment before running.",
    default=1,
)
parser.add_argument(
    "--seed", type=int, metavar="SEED", help="Seed of the experiment.", default=-1
)
parser.add_argument( #20220105
    "--gen_code", type=int, metavar="GEN_CODE", help="Type of dataset.", default=1
)

###############################################################################
# Configurations

args = parser.parse_args()

print('Gen code: ', args.gen_code) #20220105
args.name = str(args.gen_code)+'-'+args.name #20220108
print('Name: ', args.name) #20220108
#pdb.set_trace()

if args.seed == -1:
    args.seed = int(time.time())

config = dynalearn.config.ExperimentConfig.covid(
    args.name,
    args.path_to_data,
    args.path_to_best,
    args.path_to_summary,
    incidence=bool(args.incidence),
)


config.train_details.val_fraction = args.val_fraction #在figure-6/run-covid-mine.py设置val_fraction
config.train_details.val_bias = args.bias
config.dataset.bias = args.bias
config.model.lag = args.lag
config.model.lagstep = args.lagstep
if args.model != "Kapoor2020":
    config.model.gnn_name = args.model
if args.model == "FullyConnectedGNN":
    config.model.num_nodes = config.networks.num_nodes
elif args.model == "Kapoor2020":
    config.dataset.transforms = dynalearn.config.TransformConfig.kapoor2020()
    config.model = dynalearn.config.TrainableConfig.kapoor()
    config.model.lag = args.lag = 7
    config.dynamics.is_weighted = config.dynamics.is_multiplex = False
    config.model.is_weighted = config.model.is_multiplex = False
    config.model.optimizer.lr = 1e-5
    config.model.optimizer.weight_decay = 5e-4
config.model.type = args.type
config.train_details.epochs = args.epochs #在figure-6/run-covid-mine.py设置epochs
config.train_metrics = []
'''
#original
config.metrics.names = [
    "AttentionMetrics",
    "AttentionStatesNMIMetrics",
    "AttentionNodeAttrNMIMetrics",
    "AttentionEdgeAttrNMIMetrics",
    "GNNForecastMetrics",
    "VARForecastMetrics",
]
'''
config.metrics.names = [ #20211229
    #"TrueForecastMetrics",
    #"GNNForecastMetrics",
    "GNNEmbeddingMetrics"
]
config.metrics.num_steps = [1, 7, 14]

# Defining the experiment
experiment = dynalearn.experiments.Experiment(config, verbose=args.verbose) #在experiments/experiment.py的Experiment的__init__()执行self.metrics = get_metrics(config.metrics)

#experiment.clean() #original #会把之前生成的data.h5删掉 #20220104注释
experiment.begin()
experiment.dataset.setup(experiment)
experiment.save_config()

# Training on covid data
experiment.mode = "main"

'''
#original
loading_covid_data(
    experiment,
    args.path_to_covid,
    lag=args.lag,
    lagstep=args.lagstep,
    incidence=bool(args.incidence),
)

_,target = loading_covid_data( #20211229
    experiment,
    args.path_to_covid,
    lag=args.lag,
    lagstep=args.lagstep,
    gen_code=gen_code,
    incidence=bool(args.incidence),
)
'''

# only retrieve embedding # 20220115
retrieve_embedding = True
if(retrieve_embedding):
    experiment.load(label_with_mode=False) #20220115
    index = 0
    indices_dict = {}
    for i in range(experiment.dataset.data["networks"].size):
        for j in range(experiment.dataset.data["inputs"][i].size):
            indices_dict[index] = (i, j)
            index += 1
    experiment.dataset.indices = indices_dict
    experiment.test_dataset.indices = indices_dict
    experiment.val_dataset.indices = indices_dict 
    #pdb.set_trace()
    experiment.compute_metrics()
    node_embeddings = experiment.metrics["GNNEmbeddingMetrics"].data["total-1"].squeeze()[:NUM_CBGS,:]
    print('Save node embeddings?')
    pdb.set_trace()
    np.save(os.path.join(os.getcwd(),f'gt-generator/covid/outputs/node_embeddings_b{args.bias}'), node_embeddings)

    #experiment.train_model(save=False, restore_best=True,retrieve_embedding=True)
    pdb.set_trace()


_,pre_exists = loading_covid_data( #20220104
    experiment,
    args.path_to_covid,
    lag=args.lag,
    lagstep=args.lagstep,
    gen_code=args.gen_code,
    incidence=bool(args.incidence),
)

target = loading_prediction_targets( #20220104
    experiment,
    args.path_to_covid,
    lag=args.lag,
    lagstep=args.lagstep,
    gen_code=args.gen_code,
    incidence=bool(args.incidence),
    )

if(not pre_exists):
    print('Save constructed dataset first..')
    experiment.save_data(label_with_mode=False) #20220104, save constructed data



experiment.model.nn.history.reset()
experiment.callbacks[0].current_best = np.inf
#pdb.set_trace()
experiment.train_model(save=False, restore_best=True)
experiment.compute_metrics() #调用experiments/experiment.py的def compute_metrics()

#true_forecast = experiment.metrics["TrueForecastMetrics"].data["test-1"] # Ground Truth (GT) #20211229 #(20220107)这不是ground truth，是用synthetic dynamics仿真得到的结果，只适用于synthetic experiments
gnn_forecast_test = experiment.metrics["GNNForecastMetrics"].data["test-1"].squeeze()[:,:NUM_CBGS] # GNN prediction #20211229
gnn_forecast_total = experiment.metrics["GNNForecastMetrics"].data["total-1"].squeeze()[:,:NUM_CBGS] #20220109
gnn_forecast_train = experiment.metrics["GNNForecastMetrics"].data["train-1"].squeeze()[:,:NUM_CBGS] #20220109

test_days = gnn_forecast_test.shape[0]
true_test = target.squeeze()[-test_days:,:NUM_CBGS]
true_total = target.squeeze()[:,:NUM_CBGS]
true_train = target.squeeze()[:-test_days,:NUM_CBGS]

rmse_total = sqrt(mean_squared_error(gnn_forecast_total,true_total))
rmse_test = sqrt(mean_squared_error(gnn_forecast_test,true_test))
rmse_train = sqrt(mean_squared_error(gnn_forecast_train,true_train))
print('rmse_total: ', rmse_total, '\nrmse_test: ', rmse_test,'\nrmse_train: ', rmse_train)

#savepath = os.path.join(os.getcwd(), 'gt-generator/covid/outputs/gnn_forecast_results_b1.npy') #/data/chenlin/code-dynalearn/scripts/figure-6/gt-generator/covid/outputs/gnn_forecast_results.npy
#np.save(savepath, gnn_forecast_total)
#savepath = os.path.join(os.getcwd(), 'gt-generator/covid/outputs/true_results_b1.npy')
#np.save(savepath, true_total) 

#找出总病例数最多的k个CBG
#k = 10
#top_k_idx=true_total[-1,:].argsort()[::-1][0:k]

print('Top-10 CBGs with most cases: ', true_total[-1,:].argsort()[::-1][0:10])
print('Top-10 CBGs with largest prediction errors: ', abs(true_test[-1,:]-gnn_forecast_test[-1,:]).argsort()[::-1][0:10])

pdb.set_trace()

experiment.save(label_with_mode=False)
experiment.end()
experiment.zip(
    to_zip=(
        "config.pickle",
        "data.h5",
        "metrics.h5",
        "history.pickle",
        "model.pt",
        "optim.pt",
    )
)
