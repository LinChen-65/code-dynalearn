import pandas as pd
import numpy as np
import os
import pickle
import pdb

num_nodes = 52
population = np.ones(num_nodes)

'''
population = np.zeros(num_nodes)
for index in range(num_nodes):
    n = names[index]
    population[index] = pop[n]

path = f"{path_to_processed}/population.data"
'''
path = os.path.join(os.getcwd(), 'population_ones.')
with open(path, "wb") as f:
    pickle.dump(population, f)

pop = pd.read_csv(path,header=None)
print(pop.head())
pdb.set_trace()

'''
file_path = os.path.join(os.getcwd(), 'population.data')
print('file_path: ', file_path)
pdb.set_trace()

pop = pd.read_csv(file_path,header=None, sep='\s+')
print(pop.head())
'''
