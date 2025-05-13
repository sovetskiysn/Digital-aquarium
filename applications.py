import torch
import pandas as pdd # She Knows, She Knows

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
            self.stats = {'# of steps': 0,
                          '# of live agents': Agent.number_of_agents,
                          '# of foods': self.MAX_NUMBER_OF_FOODS,
                          'Avg of energy': env_parameters['START_AGENT_ENERGY'],
                          '# of Gen': 0}
            self.vis = Visualization(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS)
            self.vis.visualize_all(self.env, self.stats)

        # if excel_flag:
        #     dict_df = pdd.DataFrame(list(my_dict.items()), columns=["Параметр", "Значение"])


    def calculate_stats(self, agent_list, generation_number):
        
        fitness_list = [self.fitness_function(agent) for agent in agent_list]
        number_of_eaten_food_list = [agent.number_of_eaten_food for agent in agent_list]
        number_of_eaten_other_agent_list = [agent.number_of_eaten_other_agent for agent in agent_list]
        age_list = [agent.age for agent in agent_list]
        energy_list = [agent.energy for agent in agent_list]

        stacy = {
            'generation_number': generation_number,

            'fitness_min':np.min(fitness_list),
            'fitness_max':np.max(fitness_list),
            'fitness_average':np.average(fitness_list),
            'fitness_median':np.median(fitness_list),

            'number_of_natural_death':Agent.number_of_natural_deaths,
            'number_of_violent_death':Agent.number_of_violent_deaths,

            'eaten_food_min':np.min(number_of_eaten_food_list),
            'eaten_food_max':np.max(number_of_eaten_food_list),
            'eaten_food_average':np.average(number_of_eaten_food_list),
            'eaten_food_median':np.median(number_of_eaten_food_list),

            'eaten_other_agent_min':np.min(number_of_eaten_other_agent_list),
            'eaten_other_agent_max':np.max(number_of_eaten_other_agent_list),
            'eaten_other_agent_average':np.average(number_of_eaten_other_agent_list),
            'eaten_other_agent_median':np.median(number_of_eaten_other_agent_list),

            'age_min':np.min(age_list),
            'age_max':np.max(age_list),
            'age_average':np.average(age_list),
            'age_median':np.median(age_list),

            'energy_min':np.min(energy_list),
            'energy_max':np.max(energy_list),
            'energy_average':np.average(energy_list),
            'energy_median':np.median(energy_list),
            }
        
        return stacy

    def fitness_function(self, agent: Type[Agent]):
        return agent.age + agent.energy // 2
    

    def fitness_proportionate_selection(self, agent_list):
        
        fitness_list = [self.fitness_function(agent) for agent in agent_list]

        return agent_list, [i/sum(fitness_list) for i in fitness_list]
    

    def rank_selection(self, agent_list):
        # Оцениваем фитнес каждого агента
        fitness_list = [self.fitness_function(agent) for agent in agent_list]

        # Сортируем агентов по возрастанию фитнеса (от наихудшего к лучшему)
        sorted_agents_with_fitness = sorted(zip(agent_list, fitness_list), key=lambda x: x[1])

        # Присваиваем ранги: 1, 2, ..., N
        ranks = list(range(1, len(agent_list) + 1))

        # Вычисляем сумму рангов
        rank_sum = sum(ranks)

        # Присваиваем вероятности на основе рангов
        probabilities = [rank / rank_sum for rank in ranks]

        # Возвращаем агентов в ранжированном порядке и соответствующие вероятности
        sorted_agents = [agent for agent, _ in sorted_agents_with_fitness]
        return sorted_agents, probabilities
    

    def elitism_selection(self, agent_list, elite_count=10):
        fitness_list = [self.fitness_function(agent) for agent in agent_list]
        
        sorted_agents = [agent for agent, _ in sorted(
            zip(agent_list, fitness_list),
            key=lambda x: x[1],
            reverse=True
        )]

        elite_agents = sorted_agents[:elite_count]
        probabilities = [1.0 / elite_count] * elite_count

        return elite_agents, probabilities
    
    def uniform_crossover(self, agent_list, probabilities, new_gen_size):
        offspring_genomes = []
        for i in range(new_gen_size):
            # SELECTION
            parent1_index = np.random.choice(len(agent_list), p=probabilities)
            parent2_index = np.random.choice(len(agent_list), p=probabilities)

            parent1 = agent_list[parent1_index]
            parent2 = agent_list[parent2_index]

            # CROSSOVER
            new_genome = {}
            for key in parent1.genome:
                # Однородное скрещивание: случайный выбор веса от одного из родителей
                mask = torch.randint(0, 2, parent1.genome[key].shape, dtype=torch.bool)
                # Ставим значение из одного из родителей в зависимости от маски
                new_genome[key] = torch.where(mask, parent1.genome[key], parent2.genome[key])

            # Добавляем потомка в список
            offspring_genomes.append(new_genome)

        return offspring_genomes


    def arithmetic_crossover(self, agent_list, probabilities, new_gen_size, noise_std=0.01):
        offspring_genomes = []
        for i in range(new_gen_size):
            # SELECTION
            parent1_index = np.random.choice(len(agent_list), p=probabilities) # Особенность - такая реализация позволяет выбрать одного и того же агента как два раза, что по сути клон
            parent2_index = np.random.choice(len(agent_list), p=probabilities)

            parent1 = agent_list[parent1_index]
            parent2 = agent_list[parent2_index]

            # CROSSOVER
            new_genome = {}
            for key in parent1.genome:
                # Среднее арифметическое между весами двух родителей
                mean = (parent1.genome[key] + parent2.genome[key]) / 2
                noise = torch.randn_like(mean) * noise_std  # Нормальный шум
                new_genome[key] = mean + noise

            # Добавляем потомка в список
            offspring_genomes.append(new_genome)

        return offspring_genomes
    

    def blend_crossover(self, agent_list, probabilities, new_gen_size, alpha=0.1):
        offspring_genomes = []
        for i in range(new_gen_size):
            # SELECTION
            parent1_index = np.random.choice(len(agent_list), p=probabilities, replace=False)  # Без повторений
            parent2_index = np.random.choice(len(agent_list), p=probabilities, replace=False)  # Без повторений

            parent1 = agent_list[parent1_index]
            parent2 = agent_list[parent2_index]

            # CROSSOVER
            new_genome = {}
            for key in parent1.genome:
                # Вычисляем разницу d
                d = torch.abs(parent1.genome[key] - parent2.genome[key])

                # Определяем интервал для потомка
                lower_bound = torch.min(parent1.genome[key], parent2.genome[key]) - alpha * d
                upper_bound = torch.max(parent1.genome[key], parent2.genome[key]) + alpha * d

                offspring_value = lower_bound + (upper_bound - lower_bound) * torch.rand_like(parent1.genome[key])

                # Итоговое значение потомка для данного гена
                new_genome[key] = offspring_value

            # Добавляем потомка в список
            offspring_genomes.append(new_genome)

        return offspring_genomes

    
    def run(self):
        all_agents_list = self.env.get_specific_entities(Agent, return_type='instance_list')


        
        df = pdd.DataFrame()
        

        generation_counter = 0
        iteration_counter = 0

        fps_control_flag = True

        while generation_counter <= self.number_of_generations:
                    

            
            if self.visualization_flag:

                # Нужно чтобы работал speed slider
                self.vis.calculate_time_delta()

                # кнопки жмяк
                self.vis.process_gui_events()
                
                fps_control_flag = self.vis.get_fps_control_flag()

            
            if fps_control_flag:
                
                # Сначала выполняем 1 шаг/итерацию среды 
                agent_list_before_step = self.env.make_step() # в этом списке могут быть мертвые и уже удаленные с карты агенты

                if Agent.number_of_agents <= 0 or iteration_counter >= 10000:

                    

                    self.env.kill_all_agents()

                    statsy = self.calculate_stats(all_agents_list, generation_counter)
                    df = pdd.concat([df, pdd.DataFrame([statsy])], ignore_index=True)

                    print('\n\n\n')
                    print(f'----------generation({statsy['generation_number']}):')
                    for key,value in statsy.items():
                        print(f"{key}: {value}")




                        
                    # SELECTION
                    mating_pool, probabilities = self.fitness_proportionate_selection(all_agents_list)
                    # mating_pool, probabilities = self.rank_selection(all_agents_list)
                    # mating_pool, probabilities = self.elitism_selection(all_agents_list)


                    # CROSSOVER
                    offspring_genomes = self.uniform_crossover(mating_pool, probabilities, new_gen_size=self.POPULATION_SIZE)
                    # offspring_genomes = self.arithmetic_crossover(mating_pool, probabilities, new_gen_size=self.POPULATION_SIZE)
                    # offspring_genomes = self.blend_crossover(mating_pool, probabilities, new_gen_size=self.POPULATION_SIZE)


                    # MUTATIONS IN AGENT'S BRAIN
                    brain_class = all_agents_list[0].brain.__class__
                    offspring_genomes = [brain_class.mutate_genome(genome, percent=10) for genome in offspring_genomes]

                    
                    # REPLACEMENT
                    new_agents_brains = [brain_class(genome=genome) for genome in offspring_genomes]
                    self.env.distribute_agents_on_map(new_agents_brains)


                    all_agents_list = self.env.get_specific_entities(Agent, return_type='instance_list')


                    iteration_counter = 0
                    generation_counter += 1

                iteration_counter += 1

                agents_list = self.env.get_specific_entities(Agent, return_type='instance_list')

                self.stats['Avg of energy'] = int(np.average([agent.energy for agent in agents_list]))


            if self.visualization_flag:

                self.stats['# of steps'] = self.env.step_counter
                self.stats['# of live agents'] = Agent.number_of_agents
                self.stats['# of foods'] = Food.number_of_foods
                self.stats['# of Gen'] = generation_counter

                # рисуем парашу
                self.vis.visualize_all(self.env, self.stats)
                        

        df.to_excel("output.xlsx", index=False)


        with pdd.ExcelWriter("output.xlsx") as writer:
            df.to_excel(writer, sheet_name="Статистика", index=False)
            # dict_df.to_excel(writer, sheet_name="Статистика", index=False)









class OpenEndedApp():
    def __init__(self, env_parameters, visualization_flag=True, excel_flag=True):


        # App parameters
        self.visualization_flag = visualization_flag

        self.NUM_OF_TILE_ROWS = env_parameters['NUM_OF_TILE_ROWS']
        self.NUM_OF_TILE_COLS = env_parameters['NUM_OF_TILE_COLS']
        self.MAX_NUMBER_OF_FOODS = env_parameters['MAX_NUMBER_OF_FOODS']


        self.env = Environment(shape=(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS), parameters=env_parameters)

        self.env.generate_random_landscape()


        # Жизнеспособный геном
        coord = (self.NUM_OF_TILE_ROWS//2, self.NUM_OF_TILE_COLS//2)
        self.viable_genome = np.array([10 for i in range(64)])

        # 10% геном заменим на размножаться
        self.viable_genome[np.random.choice(self.viable_genome.size, int(self.viable_genome.size * 0.1), replace=False)] = 0

        self.env.set_agent_at_location(coord, brain=Programmatic_brain(genome=self.viable_genome))

        self.env.generate_foods(self.MAX_NUMBER_OF_FOODS)



        if visualization_flag:            
            self.stats = {'# of steps': 0,
                          '# of live agents': Agent.number_of_agents,
                          '# of foods': self.MAX_NUMBER_OF_FOODS,
                          'Avg of energy': env_parameters['START_AGENT_ENERGY'],
                          '# of Gen': 0}
            self.vis = Visualization(self.NUM_OF_TILE_ROWS, self.NUM_OF_TILE_COLS)
            self.vis.visualize_all(self.env, self.stats)


    
    def run(self):


        fps_control_flag = True

        generation_counter = 1

        while True:
                    

            
            if self.visualization_flag:

                # Нужно чтобы работал speed slider
                self.vis.calculate_time_delta()

                # кнопки жмяк
                self.vis.process_gui_events()
                
                fps_control_flag = self.vis.get_fps_control_flag()

            
            if fps_control_flag:
                
                # Сначала выполняем 1 шаг/итерацию среды 
                agent_list_before_step = self.env.make_step() # в этом списке могут быть мертвые и уже удаленные с карты агенты

                if self.env.number_of_agents <= 0:
                    
                    # print('НОВАЯ ГЕНЕРАЦИЯ')
                    self.env.distribute_agents_on_map([Programmatic_brain(genome=self.viable_genome)])
                    generation_counter += 1
                    

                agents_list = self.env.get_specific_entities(Agent, return_type='instance_list')

                self.stats['Avg of energy'] = int(np.average([agent.energy for agent in agents_list]))




            if self.visualization_flag:

                self.stats['# of steps'] = self.env.step_counter
                self.stats['# of live agents'] = Agent.number_of_agents
                self.stats['# of foods'] = Food.number_of_foods
                self.stats['# of Gen'] = generation_counter

                # рисуем парашу
                self.vis.visualize_all(self.env, self.stats)
                        




