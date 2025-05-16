from applications import *

import numpy as np
np.random.seed(42)

import torch
torch.manual_seed(42)



env_parameters = {'NUM_OF_TILE_ROWS': 30,
                  'NUM_OF_TILE_COLS': 30,
                  'POPULATION_SIZE': 30,
                  'MAX_NUMBER_OF_FOODS':100,
                  'NUMBER_OF_NEW_FOODS':10,
                  'FOOD_NUTRIATION':30,
                  'PHOTOSYNTHESIS_ENERGY':0,
                  'START_AGENT_ENERGY':30,
                  'MAX_AGENT_AGE':float('inf')}


ga_parameters = {
                'NUMBER_OF_GENERATIONS':1_000,
                'MAX_ITERATION_NUMBER': float('inf'),
                'MUTATION_PERCENTAGE': 0.01
                }



app = GeneticAlgorithmApp(env_parameters, ga_parameters, visualization_flag=False, excel_flag=True, selection_type='elitism', crossover_type='uniform')
app.run(file_name='mutation_low2.xlsx')

selection_type = ['elitism', 'rank', 'fps', 'fpes']
crossover_type = ['uniform', 'arithmetic', 'blend', 'clone']


# ============================================================


# env_parameters = {'NUM_OF_TILE_ROWS': 30,
#                   'NUM_OF_TILE_COLS': 30,
#                   'MAX_NUMBER_OF_FOODS':100,
#                   'NUMBER_OF_NEW_FOODS':10,
#                   'FOOD_NUTRIATION':30,
#                   'START_AGENT_ENERGY':30,
#                   'PHOTOSYNTHESIS_ENERGY':2,
#                   'MAX_AGENT_AGE':float('inf')}


# app = OpenEndedApp(env_parameters, visualization_flag=True)
# app.run()