import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

import sys
import os
import pickle
import numpy as np

import pdb

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants

###############################################################################
# Constants

epic_data_root = '/data/chenlin/COVID-19/Data'

###############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME) #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

###############################################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(epic_data_root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

###############################################################################
NUM_POI = poi_cbg_visits_list[0].todense().shape[0]
NUM_CBG = poi_cbg_visits_list[0].todense().shape[1]

node_list = list(np.arange(NUM_CBG))
node_attr = list(np.ones(NUM_CBG)) #test
#edge_list = list()
#edge_attr = list()