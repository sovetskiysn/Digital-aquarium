import numpy as np
from typing import Type, Literal


from engine.entities import *
from engine.brains import *


class Environment():

    def __init__(self, shape, parameters):
        
        self.world_grid = np.zeros(shape, dtype=object)

        # статистики
        self.step_counter = 0
        self.number_of_agents = 0
        self.number_of_foods = 0

        # Параметры среды
        self.max_number_of_foods = parameters['MAX_NUMBER_OF_FOODS']
        self.number_of_new_foods = parameters['NUMBER_OF_NEW_FOODS']
        self.nutriation = parameters['FOOD_NUTRIATION']
        self.start_energy_of_agent = parameters['START_AGENT_ENERGY']
        self.photosynthesis_energy = parameters['PHOTOSYNTHESIS_ENERGY']
        self.max_age = parameters['MAX_AGENT_AGE']





    def load_landscape(self,path):
        pass

    def generate_random_landscape(self):
        for row, col in np.ndindex(self.world_grid.shape):
            
            self.world_grid[row,col] = Grass((row,col), saturation_type = 'deep' if np.random.rand() < 0.25 else 'regular')
    

    def get_specific_entities(self, entity_type: Type[Entity] | tuple[Type[Entity]], return_type: Literal['instance_list', 'true_false_grid', 'coord_list'] = 'instance_list'):
        """
        Возвращает numpy array всех экземпляром или true_false_grid или лист индексов определенного типа
        """
        vectorized_func = np.vectorize(lambda entity: isinstance(entity, entity_type))
        true_false_grid = vectorized_func(self.world_grid)

        if return_type == 'instance_list':
            return self.world_grid[true_false_grid]
        elif return_type =='true_false_grid':
            return true_false_grid
        elif return_type =='coord_list':
            return np.argwhere(true_false_grid)



    def generate_foods(self, number_of_foods = 5):

        coord_list = self.get_specific_entities((Grass, Dirt), return_type='coord_list')

        diff = self.max_number_of_foods - self.number_of_foods

        number_of_foods = diff if diff<number_of_foods else number_of_foods
        
        if len(coord_list) != 0:

            indexes = np.random.choice(len(coord_list), number_of_foods if number_of_foods <= len(coord_list) else len(coord_list), replace=False)

            for row, col in coord_list[indexes]:
                self.world_grid[row,col] = Food((row,col), env=self)

            self.number_of_foods += number_of_foods


    def generate_random_agents(self, brain_class, number_of_agents = 5):

        coord_list = self.get_specific_entities((Grass, Dirt), return_type='coord_list')

        if len(coord_list) != 0:
            indexes = np.random.choice(len(coord_list), number_of_agents, replace=False)

            for row, col in coord_list[indexes]:
                self.world_grid[row,col] = Agent((row,col), brain=brain_class(), env=self)


            self.number_of_agents += number_of_agents


    def kill_all_agents(self):

        agent_list = self.get_specific_entities(Agent, return_type='instance_list')

        for agent in agent_list:
            agent.die_or_harakiri()
        


    def distribute_agents_on_map(self, brain_list):

        coord_list = self.get_specific_entities((Grass, Dirt), return_type='coord_list')

        if len(coord_list) != 0:

            indexes = np.random.choice(len(coord_list), len(brain_list), replace=False)


            for i, brain in zip(indexes, brain_list):
                
                self.set_agent_at_location(coord_list[i], brain=brain)





    def create_new_generation(self, mating_pool, new_gen_size: int):

        coord_list = self.get_specific_entities((Grass, Dirt), return_type='coord_list')
 
        if len(coord_list) != 0:


            # Индексы сводобных полей на карте
            indexes = np.random.choice(len(coord_list), new_gen_size, replace=False)
            
            # Каждому агенту присвается вероятность отсавить потомство в соответствий с его значением fitness_value
            probabilities = mating_pool[:, 0].astype(float) / mating_pool[:, 0].astype(float).sum()  
      

            for row, col in coord_list[indexes]:

                # SELECTION
                parent1_index = np.random.choice(len(mating_pool), p=probabilities)
                parent2_index = np.random.choice(len(mating_pool), p=probabilities)

                parent1 = mating_pool[parent1_index][1]
                parent2 = mating_pool[parent2_index][1]


                # CROSSOVER
                offspring_genome = {}
                for key in parent1.genome:

                    # Однородное скрещивание: случайный выбор веса от одного из родителей
                    mask = torch.randint(0, 2, parent1.genome[key].shape, dtype=torch.bool)
                    offspring_genome[key] = torch.where(mask, parent1.genome[key], parent2.genome[key])

                
                offspring_brain = parent1.brain.__class__(genome = offspring_genome)

                self.world_grid[row,col] = Agent((row, col), brain=offspring_brain, env=self)


            # тут написанно так потому что может быть предыдущее поколение жить а может быть автоматом убито
            self.number_of_agents = np.count_nonzero(self.get_specific_entities(Agent, return_type='true_false_grid'))


    def set_agent_at_location(self, coord: tuple, brain):

        row, col = coord

        if isinstance(self.world_grid[row,col], (Grass, Dirt)):

            self.world_grid[row,col] = Agent((row,col), brain=brain, env=self)

            self.number_of_agents += 1



    def clean_statistics(self):

        self.step_counter = 0
        self.number_of_agents = 0
        self.number_of_foods = 0
        self.average_energy_number = self.start_energy_of_agent


    def make_step(self):
        

    


        foods_list = self.get_specific_entities(Food, return_type='instance_list')

        for food in foods_list:
            food.update(self)


        if self.number_of_foods < self.max_number_of_foods:
            self.generate_foods(self.number_of_new_foods)



        # print('\n')
        # print(f'make_step({self.step_counter}):')

        agents_list = self.get_specific_entities(Agent, return_type='instance_list') 


        # print(f'len(agents_list) = {len(agents_list)}')
        # print(f'# of alive = {sum([int(agent.is_alive) for agent in agents_list])}')
        # print(f'self.number_of_agents = {self.number_of_agents}')
        # print(f'Agent.number_of_agents = {Agent.number_of_agents}')



        for agent in agents_list:
            agent.set_action_function()

            if agent.is_alive == False: # проверка на смерть от думания
                agent.die_or_harakiri()        
        
        # print(f'after set_action:')
        # print(f'# of alive = {sum([int(agent.is_alive) for agent in agents_list])}')
        # print(f'self.number_of_agents = {self.number_of_agents}')
        # print(f'Agent.number_of_agents = {Agent.number_of_agents}')


        # permutation нужен чтобы действия агентов выполнялись в случайном порядке 
        # случайный порядок нужен чтобы у тех кто левее не было преимущество
        stringgg = {}
        for i, agent in enumerate(np.random.permutation(agents_list)):
            
            # Нужно проверить ли живой агент, отсеет всех
            # 1) кого убил другой агент, другой агент убил текущего агента до того как он смог что-то сделать
            # 2) тех агентов которые умерли еще во время выбора действия, некоторые агенты могут умереть из-за того что потратили всю энергию во время думания
            # 3) всех тех кто умер от естественных причин

            # делаем действие если возможно
            if agent.is_alive == True: 
                agent.action_function()

                agent.age +=1
                agent.energy -= 1

                if agent.check_is_alive() == False: # проверка на естественную смерть
                    agent.die_or_harakiri()

            

        #     if hasattr(agent.action_function, '__name__'):
        #         stringgg[i] = agent.action_function.__name__
        #     else:
        #         p = agent.action_function
        #         func_name = p.func.__name__

        #             # Получить зафиксированные позиционные и ключевые аргументы
        #         fixed_args = p.args         # кортеж позиционных аргументов
        #         fixed_kwargs = p.keywords   # словарь ключевых аргументов

        #         # Собрать строку
        #         params_str = ", ".join(
        #             [repr(arg) for arg in fixed_args] +
        #             [f"{k}={v!r}" for k, v in fixed_kwargs.items()]
        #         )

        #         stringgg[i] = f"{func_name}({params_str})"


        # print(stringgg)
            
        # print(f'after action_function:')
        # print(f'# of alive = {sum([int(agent.is_alive) for agent in agents_list])}')
        # print(f'self.number_of_agents = {self.number_of_agents}')
        # print(f'Agent.number_of_agents = {Agent.number_of_agents}')        

        # for agent in agents_list:
        #     if agent.check_is_alive() == False: # случай естественной смерти
        #         agent.die_or_harakiri()
            
                

        
        # обновляем статистику
        self.step_counter += 1
        self.number_of_agents = np.count_nonzero(self.get_specific_entities(Agent, return_type='true_false_grid'))
        

        # print('------')
        # print(f'after check_is_alive:')
        # new_agents_list = self.get_specific_entities(Agent, return_type='instance_list')
        # print(f'# of alive = {sum([int(agent.is_alive) for agent in new_agents_list])}')
        # print(f'self.number_of_agents = {self.number_of_agents}')
        # print(f'Agent.number_of_agents = {Agent.number_of_agents}')
        # print(f'natural/death = {Agent.number_of_natural_deaths}/{Agent.number_of_violent_deaths}')

        

        self.number_of_foods = np.count_nonzero(self.get_specific_entities(Food, return_type='true_false_grid'))

        self.average_energy_number = sum([i.energy for i in agents_list]) // self.number_of_agents if self.number_of_agents != 0 else 0


        return agents_list


        











        

        


