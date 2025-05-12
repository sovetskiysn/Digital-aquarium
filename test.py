

from engine.environment import Environment
from engine.entities import *
from engine.brains import *

import copy
import random

import torch

torch.set_printoptions(precision=4, sci_mode=False)


def mutate_state_dict(state_dict, percent=10):

    copy_of_genome = copy.deepcopy(state_dict)


    for key in copy_of_genome:
        mask = torch.rand_like(copy_of_genome[key]) < percent * 0.01  # 0.2 = 20% probability
        copy_of_genome[key][mask] = torch.empty(mask.sum()).uniform_(-1, 1)  # Assign new random values 
    
    
    return copy_of_genome



env_parameters = {'NUM_OF_TILE_ROWS': 30,
                  'NUM_OF_TILE_COLS': 30,
                  'POPULATION_SIZE': 50,
                  'MAX_NUMBER_OF_FOODS':100,
                  'NUMBER_OF_NEW_FOODS':10,
                  'FOOD_NUTRIATION':30,
                  'START_AGENT_ENERGY':30,
                  'PHOTOSYNTHESIS_ENERGY':3,
                  'MAX_AGENT_AGE':float('inf')}

env = Environment(shape=(30, 30), parameters=env_parameters)

Ansar = Agent((-1,-1),brain=Classic_NN_brain(),env=env)
Arsen = Agent((-1,-1),brain=Classic_NN_brain(),env=env)


first = copy.deepcopy(Ansar.genome)


second = mutate_state_dict(first, percent=90)


# print(str(first) == str(second))
# print('\n\n-----------\n\n')
# print(str(first))

# print('\n\n-----------\n\n')
# print(str(second))

elite_count = 5
probabilities = [1.0 / elite_count] * elite_count

print(probabilities)