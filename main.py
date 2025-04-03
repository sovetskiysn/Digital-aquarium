import numpy as np
import pandas as pdd

from engine.environment import Environment
from engine.brains import *


from visualization.visualization import Visualization
from visualization.parameters import *



class DigitalAquariumApp():
    """
    Application модуль может быть использован как для визуализации процесса 
    так и для запусков тестов с заранее подготовленными параметрами
    """
    def __init__(self, NUM_OF_TILE_ROWS, NUM_OF_TILE_COLS):

        self.NUM_OF_TILE_ROWS = NUM_OF_TILE_ROWS
        self.NUM_OF_TILE_COLS = NUM_OF_TILE_COLS



        env_parameters = {'max_number_of_foods':180,
                          'number_of_new_foods':10,
                          'nutriation':30,
                          'start_energy':30,
                          'max_age':float('inf')}

        self.env = Environment(shape=(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS), parameters=env_parameters)
        
        self.env.generate_random_landscape()


        # Первая популяция
        self.population_size = 30
        self.env.generate_random_agents(brain_class=Classic_NN_brain, number_of_agents=self.population_size)


        self.env.generate_foods(180)

        self.vis = Visualization(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS)
        self.vis.visualize_all(self.env)





    def run(self):


        mating_pool = np.empty((self.population_size, 2), dtype=object)

        agent_list = self.env.get_specific_entities(Agent, return_type='instance_list')
        mating_pool[:, 0] = 0
        mating_pool[:, 1] = agent_list

        generation_iteration_counter = 0


        
        df = pdd.DataFrame(columns=["Generation number", "Fitness-min", "Fitness-max", "Fitness-average", "Fitness-median", 
                                    "Number of natural death", "Number of violent death", 
                                    "Number of eaten food-min", "Number of eaten food-max", "Number of eaten food-average", "Number of eaten food-median"])
        

        while self.env.number_of_generations <= 500:
                    



            # Нужно чтобы работал speed slider
            self.vis.calculate_time_delta()


            # кнопки жмяк
            self.vis.process_gui_events()
                
            # Делать шаг ENV в соответсткий с кнопкой start-stop и с нужной частотой в секунду
            if self.vis.env_playing_flag:
                self.vis.play_speed = self.vis.speed_slider.get_current_value()
                self.vis.play_speed_acc += self.vis.time_delta
                if self.vis.play_speed_acc >= self.vis.play_speed:
                    self.vis.play_speed_acc = 0.0



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
                        

                        # print(f'generation_number - {generation_number}')
                        # print(f'age_min: {age_min}')
                        # print(f'age_max: {age_max}')
                        # print(f'age_average: {age_average}')
                        # print(f'age_median: {age_median}')
                        # print(f'number_of_natural_death: {number_of_natural_death}')
                        # print(f'number_of_violent_death: {number_of_violent_death}')


                        self.env.create_new_generation(mating_pool, new_gen_size = self.population_size)


                        agent_list = self.env.get_specific_entities(Agent, return_type='instance_list')
                        mating_pool[:, 0] = 0
                        mating_pool[:, 1] = agent_list



                        self.env.number_of_generations += 1


            # рисуем парашу
            self.vis.visualize_all(self.env)
                        

        df.to_excel("output.xlsx", index=False)



            

app = DigitalAquariumApp(30,30)
app.run()