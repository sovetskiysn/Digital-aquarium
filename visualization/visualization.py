import pygame
import numpy as np
import pygame_gui
from pygame_gui.core import ObjectID
from pygame_gui.elements import UILabel, UIButton, UIHorizontalSlider, UIPanel, UIVerticalScrollBar
from typing import Literal


from engine.entities import *
from visualization.parameters import *
from visualization.textures import *





class Visualization():

    def __init__(self, NUM_OF_TILE_ROWS, NUM_OF_TILE_COLS):

        # GUI переменные
        self.time_delta = 0
        self.env_playing_flag = False
        self.play_speed = 0.3
        self.play_speed_acc = 0.0


        # General start
        pygame.init()
        pygame.display.set_caption("Evolution")
        

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()

        # GAME
        # -----------------------------------------------------------------------------------        
        self.game_rect = pygame.Rect((0,0), (WIDTH - GUI_OFFSET, HEIGHT))

        # creating and position world
        temp_width = NUM_OF_TILE_COLS * TILE_SIZE + GRID_SIZE * (NUM_OF_TILE_COLS + 1)
        temp_height = NUM_OF_TILE_ROWS * TILE_SIZE + GRID_SIZE * (NUM_OF_TILE_ROWS + 1)

        self.world_surface = pygame.Surface((temp_width, temp_height)) 
        self.world_rect = self.world_surface.get_rect(center=self.game_rect.center)



        # GUI
        # -----------------------------------------------------------------------------------
    
        self.ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT), 'visualization/ui_theme.json')

        self.ui_manager.preload_fonts([{'name': 'Segoe UI Semibold', 'point_size': 18}])

        temp_rect = pygame.Rect((WIDTH - GUI_OFFSET, 0), (GUI_OFFSET, HEIGHT))
        margin_dict = {'top':10, 'right':10, 'bottom':10, 'left':10}

        self.ui_main_panel = UIPanel(temp_rect, manager=self.ui_manager, object_id=ObjectID(class_id='MainPanel'), margins=margin_dict)
        


        # ----- BUTTONS
        width_minus_margins = GUI_OFFSET - margin_dict['left'] - margin_dict['right']
        height_minus_margins = HEIGHT - margin_dict['top'] - margin_dict['bottom']

        # ----------App Panel
        temp_rect = pygame.Rect((0, 0), (width_minus_margins, 200))
        self.app_panel = UIPanel(temp_rect, manager=self.ui_manager, container=self.ui_main_panel, object_id=ObjectID(class_id='Panel'))


        # App Panel - Main label
        temp_rect = temp_rect = pygame.Rect((0, 0), (width_minus_margins, 30))
        self.sim_panel_label = UILabel(temp_rect, "Environment", manager=self.ui_manager, container=self.app_panel, object_id = ObjectID(class_id='PanelLabel'))


        # App Panel - Steps label        
        temp_rect = pygame.Rect((10, 30), (width_minus_margins, 30))
        self.step_label = UILabel(temp_rect, "# of steps: 0", manager=self.ui_manager, container=self.app_panel, object_id = ObjectID(class_id='Label'))

        # App Panel - Number of live agents label        
        temp_rect = pygame.Rect((10, 50), (width_minus_margins, 30))
        self.number_of_agents_label = UILabel(temp_rect, "# of live agents: 0", manager=self.ui_manager, container=self.app_panel, object_id = ObjectID(class_id='Label'))

        # App Panel - Number of foods label        
        temp_rect = pygame.Rect((10, 70), (width_minus_margins, 30))
        self.number_of_foods_label = UILabel(temp_rect, "# of foods: 0", manager=self.ui_manager, container=self.app_panel, object_id = ObjectID(class_id='Label'))


        # App Panel - Average number of energy        
        temp_rect = pygame.Rect((10, 90), (width_minus_margins, 30))
        self.number_of_avg_energy_label = UILabel(temp_rect, "Avg of energy: 0", manager=self.ui_manager, container=self.app_panel, object_id = ObjectID(class_id='Label'))

        # App Panel - Number of generations       
        temp_rect = pygame.Rect((10, 110), (width_minus_margins, 30))
        self.number_of_generations = UILabel(temp_rect, "Avg of energy: 0", manager=self.ui_manager, container=self.app_panel, object_id = ObjectID(class_id='Label'))


        # -----------------Control Panel
        temp_rect = pygame.Rect((0, 165), (width_minus_margins, 250))
        self.sim_panel = UIPanel(temp_rect, manager=self.ui_manager, container=self.ui_main_panel, object_id=ObjectID(class_id='Panel'))

        # Control Panel - Main label
        temp_rect = temp_rect = pygame.Rect((0, 0), (width_minus_margins, 30))
        self.sim_panel_label = UILabel(temp_rect, "Control", manager=self.ui_manager, container=self.sim_panel, object_id = ObjectID(class_id='PanelLabel'))

        # Control Panel - Steps label
        temp_rect = temp_rect = pygame.Rect((10, 30 + 10), (width_minus_margins - 20, 30))
        self.fps_label = UILabel(temp_rect, "FPS: 0", manager=self.ui_manager, container=self.sim_panel, object_id = ObjectID(class_id='Label'))


        # Control Panel - Start-Stop button
        temp_rect = temp_rect = pygame.Rect((10, 30 + 10 + 30 + 10), (80, 40))
        self.start_stop_button = UIButton(temp_rect, text='Start', manager=self.ui_manager, container=self.sim_panel)


        # Control Panel - Speed slider
        temp_rect = pygame.Rect((10, 30 + 10 + 30 + 10 + 40 + 10), (140, 25))
        self.speed_slider = UIHorizontalSlider(temp_rect, start_value=self.play_speed, value_range=(0.6, 0.01), object_id = ObjectID(class_id='Slider'),manager=self.ui_manager, container=self.sim_panel)



    def calculate_time_delta(self):
        self.time_delta = self.clock.tick()/1000.0



    def get_fps_control_flag(self):

        if self.env_playing_flag:
            self.play_speed = self.speed_slider.get_current_value()
            self.play_speed_acc += self.time_delta
            if self.play_speed_acc >= self.play_speed:
                self.play_speed_acc = 0.0
                return True

        return False


    def process_gui_events(self):
            # Event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    exit()
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == self.start_stop_button:
                            if self.env_playing_flag:
                                self.start_stop_button.set_text('Start')
                            else:
                                self.start_stop_button.set_text('Stop')
                            self.env_playing_flag = not self.env_playing_flag


                # Buttons
                self.ui_manager.process_events(event)



    def draw_world_grid_on_world_surface(self, world_grid, visualization_mode:Literal['Agent species'] = 'Agent species'):

        self.world_surface.fill(GRID_COLOR)
        for row, col in np.ndindex(world_grid.shape):
            x = GRID_SIZE + (TILE_SIZE + GRID_SIZE) * col
            y = GRID_SIZE + (TILE_SIZE + GRID_SIZE) * row

            # Если агент стоит на локаций(траве, воде и так далее) то сначала отрисовать землю
            if hasattr(world_grid[row, col], 'under_entity_reference'):
                under_texture = self.get_texture(world_grid[row, col].under_entity_reference, texture_mode=visualization_mode)
                self.world_surface.blit(under_texture, (x, y))

            texture = self.get_texture(world_grid[row, col], texture_mode=visualization_mode)

            self.world_surface.blit(texture, (x, y))



    @staticmethod
    def get_texture(unknown_entity, texture_mode):
        """
        Метод принимает неизвестную сущность из world_grid и возвращает текстуру соответствующую этой сущности
        """

        if isinstance(unknown_entity, Grass):
            texture = Grass_texture

            if unknown_entity.saturation_type == 'deep':
                texture.fill(DEEPER_GRASS_COLOR)
            else:
                texture.fill(REGULAR_GRASS_COLOR)

        elif isinstance(unknown_entity, Dirt):
            texture = Dirt_texture

        elif isinstance(unknown_entity, Wall):
            texture = Wall_texture

        elif isinstance(unknown_entity, Agent):
            texture = Agent_texture

        elif isinstance(unknown_entity, Food):
            texture = Food_texture
        else: 
            # если это что-то неизвестное то будет базовая текстура Entity
            texture = Entity_texture

        return texture
    

    

    
    def visualize_all(self, env, stats):
    
        # ----------GUI

        # Labels
        self.fps_label.set_text(f"FPS: {round(self.clock.get_fps())}")
        self.step_label.set_text(f"# of steps: {env.step_counter}")

        self.number_of_agents_label.set_text(f"# of live agents: {env.number_of_agents}")
        self.number_of_foods_label.set_text(f"# of foods: {env.number_of_foods}")
        # self.number_of_avg_energy_label.set_text(f"Avg of energy: {stats['average_energy_number']}")

        self.number_of_generations.set_text(f"# of Gen: {stats['number_of_generations']}")



        # buttons
        self.ui_manager.update(self.time_delta)
        self.ui_manager.draw_ui(self.screen)

        
        # ---------------------Game
        # background
        pygame.draw.rect(self.screen, GAME_BACKGROUND_COLOR, self.game_rect)

        # world_grid
        self.draw_world_grid_on_world_surface(env.world_grid)

        # black border around world_grid
        pygame.draw.rect(self.screen, BORDER_COLOR, self.world_rect.inflate(BORDER_THICKNESS, BORDER_THICKNESS))

        # blit
        self.screen.blit(self.world_surface, self.world_rect)
        
        # updating screen
        pygame.display.update()

