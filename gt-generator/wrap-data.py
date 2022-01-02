# python wrap-data.py MSA_NAME gen_code
# python wrap-data.py SanFrancisco 0
 
import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

import sys
import os
import argparse
import pickle
import numpy as np

import dynalearn
import h5py

import pdb

#sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants

###############################################################################
# Constants

epic_data_root = '/data/chenlin/COVID-19/Data'
gt_result_root = os.getcwd()

###############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME) #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

gen_code = int(sys.argv[2])
if(gen_code==0):
    print('Generate data: synthetic CBG homogeneous network.')
elif(gen_code==1):
    print('Generate data: CBG-POI homogeneous network.')
else:
    print('Invalid gen_code. Please check.')
    pdb.set_trace()

###############################################################################
# Load and wrap data

# Load epidemic result data
NUM_SEEDS = 60
cases_cbg_no_vaccination = np.load(os.path.join(gt_result_root, 'cases_cbg_no_vaccination_%s_%sseeds.npy' % (MSA_NAME, NUM_SEEDS)))
print('shape of cbg daily cases: ', cases_cbg_no_vaccination.shape) #(num_days, num_cbgs)(63, 2943)
num_days = cases_cbg_no_vaccination.shape[0]; print('num_days: ', num_days)

# Load POI-CBG visiting matrices
#MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.MSA_NAME] #'San_Francisco_Oakland_Hayward_CA'
f = open(os.path.join(epic_data_root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Get one timestep's data for test 
single_array = poi_cbg_visits_list[0].todense()
num_pois = single_array.shape[0]; print('num_pois: ', num_pois)
num_cbgs = single_array.shape[1]; print('num_cbgs: ', num_cbgs)
num_nodes = num_pois + num_cbgs; print('num_nodes (cbg+poi): ', num_nodes)
#num_days = int(len(poi_cbg_visits_list)/24) #1512h/24h


# Average all arrays
num_hours = len(poi_cbg_visits_list); print('num_hours: ', num_hours)
avg_array_path = os.path.join(gt_result_root, 'avg_array_%s.npy' % MSA_NAME)
if(os.path.exists(avg_array_path)):
    avg_array = np.load(avg_array_path)
else:
    avg_array = np.zeros(poi_cbg_visits_list[0].shape)
    for i in range(num_hours):
        if(i%50==0):print(i)
        avg_array += poi_cbg_visits_list[i]
    avg_array /= num_hours
    np.save(avg_array_path, avg_array)
pdb.set_trace()

# Network attributes
#edge_list = np.append(np.reshape(np.nonzero(single_array)[0], (-1,1)),np.reshape(np.nonzero(single_array)[1], (-1,1)),axis=1)
if(gen_code==0):
    num_nodes = num_cbgs
    node_list = np.arange(num_nodes)
    node_attr = np.ones(num_nodes) #test
    #edge_list = np.append(np.random.permutation(np.reshape(np.nonzero(single_array)[1], (-1,1))),
    #                      np.reshape(np.nonzero(single_array)[1], (-1,1)),axis=1) #test
    edge_list = np.append(np.random.permutation(np.reshape(np.nonzero(avg_array)[1], (-1,1))),
                          np.reshape(np.nonzero(avg_array)[1], (-1,1)),axis=1) #20220101
elif(gen_code==1):
    cases_cbg_no_vaccination = np.concatenate((cases_cbg_no_vaccination, np.zeros((num_days,num_pois))), axis=1) #Set all poi ground truth as 0
    node_list = np.arange(num_nodes)
    node_attr = np.ones(num_nodes) #test
    edge_list = np.append(np.reshape(np.nonzero(avg_array)[0], (-1,1)),
                          np.reshape(np.nonzero(avg_array)[1]+num_cbgs, (-1,1)),axis=1)

edge_attr = np.ones(len(edge_list)) #test


# Wrap in hdf5 format
#data = h5py.File('data_%s.h5' % (MSA_NAME), 'w')
data = h5py.File('data_%s_gencode%s.h5' % (MSA_NAME, str(gen_code)), 'w')
# Epidemic data
data.create_dataset('timeseries', data=cases_cbg_no_vaccination)
# Mobility network
f = data.create_group('networks')
f.create_dataset('node_list', data=node_list)
f.create_dataset('edge_list', data=edge_list)
node_data = f.create_group('node_attr')
node_data.create_dataset('population', data=node_attr)
edge_data = f.create_group('edge_attr')
edge_data.create_dataset('weight',data=edge_attr)
networks = f

lag = 5
lagstep = 1
num_states = 1

# Wrap data
'''
#dataset = h5py.File(os.path.join(path_to_covid, "spain-covid19cases.h5"), "r")
print('path_to_covid: ', path_to_covid)
dataset = h5py.File(os.path.join(os.path.abspath('../..'),path_to_covid, "spain-covid19-dataset.h5"), "r") #20211221
num_states = 1

X = dataset["weighted-multiplex/data/timeseries/d0"][...] #20211221
Y = dataset["weighted-multiplex/data/timeseries/d0"][...] #20211221
networks = dataset["weighted-multiplex/data/networks/d0"]
'''

data = {
    "inputs": dynalearn.datasets.DataCollection(name="inputs"),
    "targets": dynalearn.datasets.DataCollection(name="targets"),
    "networks": dynalearn.datasets.DataCollection(name="networks"),
}
inputs = np.zeros((num_days - (lag - 1) * lagstep, num_nodes, num_states, lag))
targets = np.zeros((num_days - (lag - 1) * lagstep, num_nodes, num_states))
X = cases_cbg_no_vaccination
Y = cases_cbg_no_vaccination
for t in range(inputs.shape[0]):
    x = X[t : t + lag * lagstep : lagstep]
    y = Y[t + lag * lagstep - 1]
    x = x.reshape(*x.shape, 1)
    y = y.reshape(*y.shape, 1)
    x = np.transpose(x, (1, 2, 0))
    inputs[t] = x
    targets[t] = y

data["inputs"].add(dynalearn.datasets.StateData(data=inputs))
data["targets"].add(dynalearn.datasets.StateData(data=targets))
data["networks"].add(dynalearn.datasets.NetworkData(data=networks))

pdb.set_trace()
