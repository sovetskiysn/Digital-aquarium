from applications import *



env_parameters = {'NUM_OF_TILE_ROWS': 30,
                  'NUM_OF_TILE_COLS': 30,
                  'POPULATION_SIZE': 50,
                  'MAX_NUMBER_OF_FOODS':100,
                  'NUMBER_OF_NEW_FOODS':10,
                  'FOOD_NUTRIATION':30,
                  'START_AGENT_ENERGY':30,
                  'PHOTOSYNTHESIS_ENERGY':3,
                  'MAX_AGENT_AGE':float('inf')}


ga_parameters = {'NUMBER_OF_GENERATIONS':10,
                'MAX_ITERATION_NUMBER':10_000}



app = GeneticAlgorithmApp(env_parameters, ga_parameters, visualization_flag=True)
app.run()



# env_parameters = {'NUM_OF_TILE_ROWS': 30,
#                   'NUM_OF_TILE_COLS': 30,
#                   'MAX_NUMBER_OF_FOODS':100,
#                   'NUMBER_OF_NEW_FOODS':10,
#                   'FOOD_NUTRIATION':30,
#                   'START_AGENT_ENERGY':30,
#                   'PHOTOSYNTHESIS_ENERGY':3,
#                   'MAX_AGENT_AGE':float('inf')}


# app = OpenEndedApp(env_parameters, visualization_flag=True)
# app.run()