# from engine.environment import Environment
# from engine.entities import *
# from engine.brains import *

# import copy
# import random

# import torch

# torch.set_printoptions(precision=4, sci_mode=False)


# def uniform_crossover(agent_list, probabilities, new_gen_size):
#     brains = []
#     for i in range(new_gen_size):
#         # SELECTION
#         parent1_index = np.random.choice(len(agent_list), p=probabilities)
#         parent2_index = np.random.choice(len(agent_list), p=probabilities)

#         parent1 = agent_list[parent1_index]
#         parent2 = agent_list[parent2_index]

#         # CROSSOVER
#         offspring_genome = {}
#         for key in parent1.genome:
#             # Однородное скрещивание: случайный выбор веса от одного из родителей
#             mask = torch.randint(0, 2, parent1.genome[key].shape, dtype=torch.bool)
#             # Ставим значение из одного из родителей в зависимости от маски
#             offspring_genome[key] = torch.where(mask, parent1.genome[key], parent2.genome[key])

#         # Создаем потомка
#         offspring_brain = parent1.brain.__class__(genome=offspring_genome)

#         # Добавляем потомка в список
#         brains.append(offspring_brain)

#     return brains



# def blend_crossover(agent_list, probabilities, new_gen_size, alpha=0.1):
#     brains = []
#     for i in range(new_gen_size):
#         # SELECTION
#         parent1_index = np.random.choice(len(agent_list), p=probabilities, replace=False)  # Без повторений
#         parent2_index = np.random.choice(len(agent_list), p=probabilities, replace=False)  # Без повторений

#         parent1 = agent_list[parent1_index]
#         parent2 = agent_list[parent2_index]

#         # CROSSOVER
#         offspring_genome = {}
#         for key in parent1.genome:
#             # Вычисляем разницу d
#             d = torch.abs(parent1.genome[key] - parent2.genome[key])

#             # Определяем интервал для потомка
#             lower_bound = torch.min(parent1.genome[key], parent2.genome[key]) - alpha * d
#             upper_bound = torch.max(parent1.genome[key], parent2.genome[key]) + alpha * d

#             offspring_value = lower_bound + (upper_bound - lower_bound) * torch.rand_like(parent1.genome[key])

#             # Итоговое значение потомка для данного гена
#             offspring_genome[key] = offspring_value

#         # Создаем потомка
#         offspring_brain = parent1.brain.__class__(genome=offspring_genome)

#         # Добавляем потомка в список
#         brains.append(offspring_brain)

#     return brains

# env_parameters = {'NUM_OF_TILE_ROWS': 30,
#                   'NUM_OF_TILE_COLS': 30,
#                   'POPULATION_SIZE': 50,
#                   'MAX_NUMBER_OF_FOODS':100,
#                   'NUMBER_OF_NEW_FOODS':10,
#                   'FOOD_NUTRIATION':30,
#                   'START_AGENT_ENERGY':30,
#                   'PHOTOSYNTHESIS_ENERGY':3,
#                   'MAX_AGENT_AGE':float('inf')}

# env = Environment(shape=(30, 30), parameters=env_parameters)

# Ansar = Agent((-1,-1),brain=Micro_NN_brain(),env=env)
# Arsen = Agent((-1,-1),brain=Micro_NN_brain(),env=env)


# brain_list = blend_crossover([Ansar,Arsen],[0.5,0.5], new_gen_size=3)

# first = Agent((-1,-1,), brain=brain_list[0], env=env)
# second = Agent((-1,-1,), brain=brain_list[1], env=env)
# third = Agent((-1,-1,), brain=brain_list[2], env=env)

# print('Ansar.genome:')
# print(str(Ansar.genome))
# print('\n------------\n')
# print('Arsen.genome:')
# print(str(Arsen.genome))

# print('\n\n===================================\n\n')

# print('first.genome:')
# print(str(first.genome))
# print('\n------------\n')
# print('second.genome:')
# print(str(second.genome))
# print('\n------------\n')
# print('third.genome:')
# print(str(third.genome))
import copy
import numpy as np

class MyClass:
    def my_method(self):
        print("Hello")

    def mutate_genome(genome, percent=100):

        mutated = genome

        length_of_genome = len(genome)
        indexes = [i for i in range(length_of_genome)]
        max_number_of_mutation = round( length_of_genome * percent / 100)

        mutation_indexes = np.random.choice(indexes, size=max_number_of_mutation, replace=False)

        for i in mutation_indexes:
            mutated[i] = np.random.randint(0, length_of_genome-1)

        return mutated





viable_genome = np.array([10 for i in range(16)])

# 10% геном заменим на размножаться
viable_genome[np.random.choice(viable_genome.size, int(viable_genome.size * 0.1), replace=False)] = 0


print(viable_genome)

new_genome = MyClass.mutate_genome(genome=viable_genome)

print(new_genome)
