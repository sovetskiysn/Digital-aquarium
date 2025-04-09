import numpy as np
import pandas as pdd

from engine.environment import Environment
from engine.brains import *


from visualization.visualization import Visualization
from visualization.parameters import *


# В этом файле я планирую написать разные классыи приложений потому-что иногда мне надо запускать open-ended, 
# иногда Genetic algorithm с каким-то определенным Selection, в другой момент GA с другим Selection





class GeneticAlgorithmApp():
    def __init__(self, env_parameters, ga_parameters, visualization_flag=True, excel_flag=True):


        # App parameters
        self.visualization_flag = visualization_flag

        self.NUM_OF_TILE_ROWS = env_parameters['NUM_OF_TILE_ROWS']
        self.NUM_OF_TILE_COLS = env_parameters['NUM_OF_TILE_COLS']
        self.POPULATION_SIZE = env_parameters['POPULATION_SIZE']
        self.MAX_NUMBER_OF_FOODS = env_parameters['MAX_NUMBER_OF_FOODS']

        # GA parameters
        self.number_of_generations = ga_parameters['NUMBER_OF_GENERATIONS']
        self.max_iteration_number = ga_parameters['MAX_ITERATION_NUMBER']

        self.env = Environment(shape=(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS), parameters=env_parameters)

        self.env.generate_random_landscape()


        # Первая популяция
        self.env.generate_random_agents(brain_class=Classic_NN_brain, number_of_agents=self.POPULATION_SIZE)
        self.env.generate_foods(self.MAX_NUMBER_OF_FOODS)



        if visualization_flag:
            self.vis = Visualization(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS)
            self.vis.visualize_all(self.env)

    
    def run(self):
        

        mating_pool = np.empty((self.POPULATION_SIZE, 2), dtype=object)

        agent_list = self.env.get_specific_entities(Agent, return_type='instance_list')
        mating_pool[:, 0] = 0
        mating_pool[:, 1] = agent_list

        generation_iteration_counter = 0


        
        df = pdd.DataFrame(columns=["Generation number", "Fitness-min", "Fitness-max", "Fitness-average", "Fitness-median", 
                                    "Number of natural death", "Number of violent death", 
                                    "Number of eaten food-min", "Number of eaten food-max", "Number of eaten food-average", "Number of eaten food-median"])
        


        fps_control_flag = True

        while self.env.number_of_generations <= 100:
                    

            
            if self.visualization_flag:

                # Нужно чтобы работал speed slider
                self.vis.calculate_time_delta()

                # кнопки жмяк
                self.vis.process_gui_events()
                
                fps_control_flag = self.vis.get_fps_control_flag()

            
            if fps_control_flag:
                
                # Сначала выполняем 1 шаг/итерацию среды 
                agent_list_after_performing_action = self.env.make_step() # в этом списке могут быть мертвые и уже удаленные с карты агенты




                generation_iteration_counter += 1

                # print(f'generation_iteration_counter - {generation_iteration_counter}')


                if self.env.number_of_agents <= 0 or generation_iteration_counter >= 10000:

                    self.env.kill_all_agents()

                    generation_iteration_counter = 0

                    

                    for agent in mating_pool[:, 1]:

                        fitness_value = agent.age

                        # если агент умер то добавляем количество энергий которое было у него при смерти
                        if agent.is_alive == False:
                            fitness_value += agent.energy // 2

                        
                        # находим агента в mating_pool
                        index = np.where([agentj is agent for agentj in mating_pool[:, 1]])[0]

                        # обновляем его значение fitness
                        mating_pool[index, 0] = fitness_value


                    generation_number = self.env.number_of_generations


                    fitness_min = np.min(mating_pool[:, 0])
                    fitness_max = np.max(mating_pool[:, 0])
                    fitness_average = np.average(mating_pool[:, 0])
                    fitness_median = np.median(mating_pool[:, 0])


                    number_of_natural_death = sum([int(agent.natural_death) for agent in mating_pool[:, 1]])
                    number_of_violent_death = sum([int(agent.be_eaten) for agent in mating_pool[:, 1]])


                    number_of_eaten_food_list = [agent.number_of_eaten_food for agent in mating_pool[:, 1]]
                    number_of_eaten_food_min = np.min(number_of_eaten_food_list)
                    number_of_eaten_food_max = np.max(number_of_eaten_food_list)
                    number_of_eaten_food_average = np.average(number_of_eaten_food_list)
                    number_of_eaten_food_median = np.median(number_of_eaten_food_list)


                    age_list = [agent.age for agent in mating_pool[:, 1]]
                    age_min = np.min(age_list)
                    age_max = np.max(age_list)
                    age_average = np.average(age_list)
                    age_median = np.median(age_list)



                    df.loc[len(df)] = [generation_number, fitness_min, fitness_max, fitness_average, fitness_median, 
                                number_of_natural_death, number_of_violent_death,
                                number_of_eaten_food_min, number_of_eaten_food_max, number_of_eaten_food_average, number_of_eaten_food_median]


                    print([generation_number, fitness_min, fitness_max, fitness_average, fitness_median, 
                                number_of_natural_death, number_of_violent_death,
                                number_of_eaten_food_min, number_of_eaten_food_max, number_of_eaten_food_average, number_of_eaten_food_median])


                    self.env.create_new_generation(mating_pool, new_gen_size = self.POPULATION_SIZE)


                    agent_list = self.env.get_specific_entities(Agent, return_type='instance_list')
                    mating_pool[:, 0] = 0
                    mating_pool[:, 1] = agent_list



                    self.env.number_of_generations += 1

            if self.visualization_flag:
                # рисуем парашу
                self.vis.visualize_all(self.env)
                        

        df.to_excel("output.xlsx", index=False)








class OpenEndedApp():
    def __init__(self, parameters):

        self.NUM_OF_TILE_ROWS = parameters['NUM_OF_TILE_ROWS']
        self.NUM_OF_TILE_COLS = parameters['NUM_OF_TILE_COLS']

        self.POPULATION_SIZE = parameters['POPULATION_SIZE']
        self.MAX_NUMBER_OF_FOODS = parameters['MAX_NUMBER_OF_FOODS']

        self.env = Environment(shape=(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS), parameters=parameters)

        self.env.generate_random_landscape()


        # Первая популяция
        self.env.generate_random_agents(brain_class=Classic_NN_brain, number_of_agents=self.POPULATION_SIZE)
        self.env.generate_foods(self.MAX_NUMBER_OF_FOODS)



        self.vis = Visualization(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS)
        self.vis.visualize_all(self.env)

    
    def run(self, visualization_flag=True):
        pass


