import numpy as np
from typing import Type, Literal

from functools import partial
import copy

import torch
import torch.nn as nn

from engine.entities import *


# class Basic_Brain:
#     def __init__(self, genome = None):

#         self.genome = genome if genome is not None else self.generate_random_genome()

#     def get_the_chosen_action(self, agent):
#         pass

#     def get_copy_of_genome(self):
#         pass
    

class Programmatic_brain:

    # Геном это та вещь которая будет подвержина мутациям и будет передаваться по наследству
    # Мозг это та вещь которая помогает боту выбрать действие, это как обработчик генома

    def __init__(self, genome = None, length_of_genome = 64):

        self.counter = 0


        self.length_of_genome = len(genome) if genome is not None else length_of_genome
        # genome это umpy array
        self.genome = genome if genome is not None else self.generate_random_genome()

    def generate_random_genome(self):
        return np.array([np.random.randint(0, self.length_of_genome-1) for i in range(self.length_of_genome)])
    
    def move_the_counter(self, shift):
        self.counter = (self.counter + shift) % self.length_of_genome

    def get_the_chosen_action(self, agent: Type[Agent]):

        
        action_function = agent.do_nothing



        while agent.check_is_alive() == True:

            command = self.genome[self.counter]

            # Размножиться
            if command == 0:        
                action_function = agent.reproduce_if_possible
                self.move_the_counter(1)
                break

            # Move and bite absolutely
            elif 1 <= command <= 8:        
                action_function = partial(agent.move_and_bite_if_possible, direction=command)
                self.move_the_counter(1)
                break

            # Move and bite relatively
            elif command == 9:
                next_command = self.genome[(self.counter + 1) % self.length_of_genome]
                action_function = partial(agent.move_and_bite_if_possible, direction = (next_command % 8) + 1)
                self.move_the_counter(2)
                break

            elif command == 10:
                action_function = agent.photosynthesis
                self.move_the_counter(1)
                break

            else:


                self.move_the_counter(command)

                # Штраф за прокрутку
                agent.energy -= 1

                
           
        return action_function


    def get_surounding_of_agent(self, agent, return_type:Literal['all', 'agents', 'foods', 'relatives'] = 'all', distance = 1):

        agent.get_surrounding_locations((agent.row, agent.col), Food)

    def get_agent_energy(self, world_grid, agent):
        return agent.energy

    def get_agent_age(self, world_grid):
        pass

    def get_number_of_agent_kills(self, world_grid):
        pass

    def get_previous_performed_action(self, world_grid):
        pass

    def get_copy_of_genome(self):
        return copy.deepcopy(self.genome)
    
    def mutate_genome(genome, percent=10):

        mutated = copy.deepcopy(genome)

        length_of_genome = len(genome)
        indexes = [i for i in range(length_of_genome)]
        max_number_of_mutation = round( length_of_genome * percent / 100)

        mutation_indexes = np.random.choice(indexes, size=max_number_of_mutation, replace=False)

        for i in mutation_indexes:
            mutated[i] = np.random.randint(0, length_of_genome-1)

        return mutated







class Forward_NN_brain(nn.Module):

    def __init__(self, genome = None):

        super(Forward_NN_brain, self).__init__()

        
        # Входные параметры это 2 блока вокруг агента плюс Количество энергий, Текущий возраст

        # [*][*][*][*][*]  - where A = agent
        # [*][*][*][*][*]  - Другой агент=2
        # [*][*][A][*][*]  - Трава=0
        # [*][*][*][*][*]  - Еда=1
        # [*][*][*][*][*]  - Стена = -1 

        self.HidenLayer1 = nn.Linear(25 + 2, 10)
        self.HidenLayer2 = nn.Linear(10, 10)
        self.HidenLayer3 = nn.Linear(10, 10)
        self.OutputLayer = nn.Linear(10, 8)

        # на выходе 17 действий который может сделать агент. Ходить в одном из 8 направлеений, Атаковать в одном из 8 направлений
        self.genome = self.set_genome(genome) if genome is not None else self.generate_random_genome()


    def forward(self, x):
        # x is observation
        x = torch.from_numpy(x).type(torch.FloatTensor)
        x = torch.relu(self.HidenLayer1(x))
        x = torch.relu(self.HidenLayer2(x))
        x = torch.relu(self.HidenLayer3(x))
        x = torch.softmax(self.OutputLayer(x), dim=-1)

        return x
    
    def set_genome(self, genome):

        self.load_state_dict(genome)
        return self.state_dict()
    

    def generate_random_genome(self):

        torch.nn.init.kaiming_uniform_(self.HidenLayer1.weight, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.HidenLayer1.bias, 0)
        torch.nn.init.kaiming_uniform_(self.HidenLayer2.weight, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.HidenLayer2.bias, 0)
        torch.nn.init.kaiming_uniform_(self.HidenLayer3.weight, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.HidenLayer3.bias, 0)
        torch.nn.init.kaiming_uniform_(self.OutputLayer.weight, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.OutputLayer.bias, 0)

        
        return self.state_dict()
        

    def create_observation(self, agent: Type[Agent]):

        replace_dict = {Food: 2,
                        Agent: 1, 
                        Grass: 0,
                        Dirt: 0, 
                        Wall: -1}

        replace_func = np.vectorize(lambda obj: replace_dict.get(type(obj), 10))       # .get(keyname, default)

        agent_surroundings = agent.get_surrounding_locations(Entity, distance=2, wall_border=True, return_type='instance_list')

        agent_surroundings = replace_func(agent_surroundings)

        agent_surroundings = agent_surroundings.reshape(-1)

        return np.append(agent_surroundings, [agent.energy, agent.age])

        
        

    def get_the_chosen_action(self, agent: Type[Agent]):
        
        x = self.create_observation(agent)

        output = self.forward(x)

        action = output.argmax().item()

        action_function = None


        if 0 <= action <= 7: # move and bite
            action_function = partial(agent.move_and_bite_if_possible, direction=action + 1)
        else:
            print('ERROR')

        return action_function

    def get_copy_of_genome(self):
        return copy.deepcopy(self.genome)
    

    def mutate_genome(genome, percent=10, bounds=(-3,3)):

        mutated = copy.deepcopy(genome)

        for key in mutated:
            mask = torch.rand_like(mutated[key]) < (percent / 100)  # percent is value in 0% - 100%
            mutated[key][mask] = torch.empty(mask.sum()).uniform_(bounds[0], bounds[1])  # Assign new random values 
    

        return mutated



class Micro_NN_brain(nn.Module):

    def __init__(self, genome = None):

        super(Micro_NN_brain, self).__init__()

        
        # Входные параметры это 2 блока вокруг агента плюс Количество энергий, Текущий возраст

        # [*][*][*][*][*]  - where A = agent
        # [*][*][*][*][*]  - Другой агент=2
        # [*][*][A][*][*]  - Трава=0
        # [*][*][*][*][*]  - Еда=1
        # [*][*][*][*][*]  - Стена = -1 

        self.HidenLayer1 = nn.Linear(9+2 , 10)
        self.OutputLayer = nn.Linear(10, 8)

        # на выходе 17 действий который может сделать агент. Ходить в одном из 8 направлеений, Атаковать в одном из 8 направлений
        self.genome = self.set_genome(genome) if genome is not None else self.generate_random_genome()


    def forward(self, x):
        # x is observation
        x = torch.from_numpy(x).type(torch.FloatTensor)
        x = torch.relu(self.HidenLayer1(x))
        x = torch.softmax(self.OutputLayer(x), dim=-1)

        return x
    
    def set_genome(self, genome):

        self.load_state_dict(genome)
        return self.state_dict()
    

    def generate_random_genome(self):

        torch.nn.init.kaiming_uniform_(self.HidenLayer1.weight, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.HidenLayer1.bias, 0)
        torch.nn.init.kaiming_uniform_(self.OutputLayer.weight, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.OutputLayer.bias, 0)

        
        return self.state_dict()
        

    def create_observation(self, agent: Type[Agent]):

        replace_dict = {Food: 2,
                        Agent: 1, 
                        Grass: 0,
                        Dirt: 0, 
                        Wall: -1}

        replace_func = np.vectorize(lambda obj: replace_dict.get(type(obj), 10))       # .get(keyname, default)

        agent_surroundings = agent.get_surrounding_locations(Entity, distance=1, wall_border=True, return_type='instance_list')

        agent_surroundings = replace_func(agent_surroundings)

        agent_surroundings = agent_surroundings.reshape(-1)

        return np.append(agent_surroundings, [agent.energy, agent.age])

        
        

    def get_the_chosen_action(self, agent: Type[Agent]):
        
        x = self.create_observation(agent)

        output = self.forward(x)

        action = output.argmax().item()

        action_function = None


        if 0 <= action <= 7: # move and bite
            action_function = partial(agent.move_and_bite_if_possible, direction=action + 1)
        else:
            print('ERROR')

        return action_function

    def get_copy_of_genome(self):
        return copy.deepcopy(self.genome)
    

    def mutate_genome(genome, percent=10, bounds=(-3,3)):

        mutated = copy.deepcopy(genome)

        for key in mutated:
            mask = torch.rand_like(mutated[key]) < (percent / 100)  # percent is value in 0% - 100%
            mutated[key][mask] = torch.empty(mask.sum()).uniform_(bounds[0], bounds[1])  # Assign new random values 
    

        return mutated






