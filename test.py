from engine.entities import *
from engine.brains import *
from engine.environment import Environment


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

env.generate_random_landscape()

Artem = Agent((-1,-1), brain=Programmatic_brain(), env=env)


print(Artem.energy)
print(Artem.age)
print(Artem.is_alive)

# death
Artem.is_alive = False
print('----')

print(Artem.check_is_alive())

print(Artem.energy)
print(Artem.age)
print(Artem.is_alive)
