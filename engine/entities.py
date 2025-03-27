import numpy as np
from typing import Type, Literal, Optional


class Entity:

    def __init__(self, coord: tuple):
        self.row, self.col = coord
            
    @property
    def coord(self):
        return (self.row, self.col)
    
    @coord.setter
    def coord(self, new):
        self.row, self.col = new
        

class Grass(Entity):
    def __init__(self, coord: tuple, saturation_type:Literal['deep', 'regular'] = 'regular'):
        super().__init__(coord)

        self.saturation_type = saturation_type




class Dirt(Entity):
    def __init__(self, coord: tuple):
        super().__init__(coord)


class Wall(Entity):
    def __init__(self, coord: tuple):
        super().__init__(coord)



class Food(Entity):

    def __init__(self, coord: tuple, env):
        super().__init__(coord)

        self.under_entity_reference = env.world_grid[self.row,self.col]
        self.nutriation = env.nutriation

        self.age = 0

    def got_bitten(self):
        return self.nutriation
    

    def update(self, env_link):
        
        self.age += 1

        if self.age >= 100:
            env_link.world_grid[self.row,self.col] = self.under_entity_reference





class Agent(Entity):

    # Этот класс используется как прослойка между мозгом и средой. Основная логика действий содержится в мозге
    # а класс Agent нужен как прослойка через которую мозг будет взаимодействовать с средой

    # [1][2][3]
    # [8][A][4]  - directions, where A = agent
    # [7][6][5]


    offsets = {
        1: (-1, -1),
        2: (-1, 0),
        3: (-1, 1),
        4: (0, 1),
        5: (1, 1),
        6: (1, 0),
        7: (1, -1),
        8: (0, -1)
    }


    def __init__(self, coord: tuple ,brain, env):
        super().__init__(coord)

        self.brain = brain

        self.env_link = env



        # Сущность на котором стоит агент
        self.under_entity_reference = env.world_grid[self.row,self.col]

        # свойство которое будет функцией действия
        self.action_function = None


        # Agent features
        self.energy = env.start_energy_of_agent
        self.max_age = env.max_age


        self.number_of_eaten_food = 0
        self.number_of_eaten_other_agent = 0


        self.natural_death = False
        self.be_eaten = False

        
        self.age = 0
        self.is_alive = True



    @property
    def genome(self):
        return self.brain.genome
        



    def set_action_function(self):        

        # на этот момент агент просто определил какое действие ему хочется сделать
        self.action_function = self.brain.get_the_chosen_action(self)


        


    
    def get_surrounding_locations(self, entity_type: Type[Entity] | tuple[Type[Entity]], distance:int = 1, wall_border:bool = False, return_type: Literal['instance_list', 'true_false_grid', 'coord_list'] = 'instance_list'):


        row, col = self.coord

        # add a wrap if it necessary
        if wall_border==True:

            world_grid_link = np.pad(self.env_link.world_grid, pad_width=distance, mode='constant', constant_values=Wall((-1,-1)))
            row += distance
            col += distance
            # Это нужно на тот случай когда агент на краю world_grid, чтобы он видел все что за границей как стену
            # Это особенная стена она только для границ Мира

        else:
            world_grid_link = self.env_link.world_grid
            # В этом случаем агент получает обрезанную матрицу то есть область видимости уменьшается у края world_grid

        
        # to crop the world_grid
        row_number, col_number = world_grid_link.shape

        top_boundary = max(0, row - distance)
        bottom_boundary = min(row_number, row + distance + 1)

        left_boundary = max(0, col - distance)
        right_boundary = min(col_number, col + distance + 1)

        croped_world_grid = world_grid_link[top_boundary:bottom_boundary,left_boundary:right_boundary]

        # function
        vectorized_func = np.vectorize(lambda entity: isinstance(entity, entity_type))
        true_false_grid = vectorized_func(croped_world_grid)
        
        if return_type == 'instance_list':
        
            return croped_world_grid[true_false_grid]

        elif return_type == 'true_false_grid':

            return true_false_grid
        
        elif return_type == 'coord_list':

            indexes = np.argwhere(true_false_grid)

            indexes[:, 0] += int((top_boundary + bottom_boundary-1) / 2)
            indexes[:, 1] += int((left_boundary + right_boundary-1) / 2)

            return indexes






    def reproduce_if_possible(self):

        available_locations_list = self.get_surrounding_locations((Grass, Dirt), return_type='coord_list')

        

        if len(available_locations_list) != 0 and self.energy <= self.env_link.start_energy_of_agent:

            random_index = np.random.choice(len(available_locations_list), 1, replace=False)

            target_row = available_locations_list[random_index][0][0]
            target_col = available_locations_list[random_index][0][1]

            

            # Создаем новый мозг для этого агента с возможными мутациями
            descendent_brain = self.brain.__class__(genome = self.brain.get_copy_of_genome())

            # Создаем оболочку Агента
            self.env_link.world_grid[target_row, target_col] = Agent((target_row, target_col), brain=descendent_brain, env=self.env_link)



            self.energy //= 2

        else:
            self.die_or_harakiri()

    
    def bite_if_possible(self, direction: int):

        target_row = self.row + self.offsets[direction][0]
        target_col = self.col + self.offsets[direction][1]

        # Проверка на то что внутри ли мира хочет сделать действие
        target_location_correctness = 0 <= target_row < self.env_link.world_grid.shape[0] and 0 <= target_col < self.env_link.world_grid.shape[1]

        if target_location_correctness and isinstance(self.env_link.world_grid[target_row, target_col], (Agent, Food)):



            # Убиваем агента добычу
            prey_entity = self.env_link.world_grid[target_row, target_col]

            # жертва принимает свлю смерть
            self.energy += prey_entity.got_bitten()


            # ставим агента в новое место
            self.env_link.world_grid[target_row, target_col] = self
            
            # в его старом месте заменяем агента на текстуру под ним в его старом месте
            self.env_link.world_grid[self.row, self.col] = self.under_entity_reference

            # Отбираем локацию под агентом у агента добычи
            self.under_entity_reference = prey_entity.under_entity_reference

            # обновляем координаты агента
            self.row = target_row
            self.col = target_col


    def photosynthesis(self):
        self.energy += self.env_link.photosynthesis_energy


    def move_if_possible(self, direction: int):

        target_row = self.row + self.offsets[direction][0]
        target_col = self.col + self.offsets[direction][1]

        # Проверка на то что внутри ли мира хочет сделать действие.
        target_location_correctness = 0 <= target_row < self.env_link.world_grid.shape[0] and 0 <= target_col < self.env_link.world_grid.shape[1]

        if target_location_correctness and not isinstance(self.env_link.world_grid[target_row, target_col], (Agent, Wall, Food)):


            # Сохраняем будущую локацию
            target_location_entity = self.env_link.world_grid[target_row, target_col]

            # ставим агента в новое место
            self.env_link.world_grid[target_row, target_col] = self
            
            # заменяем агента на текстуру под ним в его старом месте
            self.env_link.world_grid[self.row, self.col] = self.under_entity_reference
            
            # обновляем локацию у агента на новую
            self.under_entity_reference = target_location_entity

            # обновляем координаты агента
            self.row = target_row
            self.col = target_col 


    def move_and_bite_if_possible(self, direction: int):

        target_row = self.row + self.offsets[direction][0]
        target_col = self.col + self.offsets[direction][1]

        # Проверка на то что внутри ли мира хочет сделать действие
        target_location_correctness = 0 <= target_row < self.env_link.world_grid.shape[0] and 0 <= target_col < self.env_link.world_grid.shape[1]

        if target_location_correctness and not isinstance(self.env_link.world_grid[target_row, target_col], Wall):


            if isinstance(self.env_link.world_grid[target_row, target_col], Agent):
                self.number_of_eaten_other_agent += 1
            elif isinstance(self.env_link.world_grid[target_row, target_col], Food):
                self.number_of_eaten_food += 1

            # Сохроняем сущность добычу
            target_location_entity = self.env_link.world_grid[target_row, target_col]

            # если эту сущность можно укусить, то она принимает свою смерть
            if hasattr(target_location_entity, 'got_bitten') and callable(target_location_entity.got_bitten):
                self.energy += target_location_entity.got_bitten()


            # ставим агента в новое место
            self.env_link.world_grid[target_row, target_col] = self
            
            # в его старом месте заменяем агента на текстуру под ним в его старом месте
            self.env_link.world_grid[self.row, self.col] = self.under_entity_reference

            if hasattr(target_location_entity, 'under_entity_reference'):
                # Отбираем локацию под агентом у агента добычи
                self.under_entity_reference = target_location_entity.under_entity_reference
            else:
                self.under_entity_reference = target_location_entity

            # обновляем координаты агента
            self.row = target_row
            self.col = target_col



    def check_is_alive(self):

        self.is_alive = (self.energy > 0 and self.age <= self.max_age) and self.is_alive == True

        if (self.energy > 0 and self.age <= self.max_age) == False:

            self.natural_death = True

        return self.is_alive
    
    def die_or_harakiri(self):
        self.is_alive = False
        self.env_link.world_grid[self.row,self.col] = self.under_entity_reference

    def got_bitten(self):
        self.is_alive = False
        self.be_eaten = True
        return self.energy//2




