import pygame
from visualization.parameters import *



# Entity
ENTITY_COLOR = '#808080'

Entity_texture = pygame.Surface((TILE_SIZE, TILE_SIZE))
Entity_texture.fill(ENTITY_COLOR)        




# WALL
WALL_COLOR = '#5f5f5f'
WALL_LINE_COLOR = '#000000'
WALL_THICKNESS = 2

Wall_texture = pygame.Surface((TILE_SIZE, TILE_SIZE))
Wall_texture.fill(WALL_COLOR)
pygame.draw.rect(Wall_texture, WALL_LINE_COLOR, (0, 0, TILE_SIZE, TILE_SIZE), WALL_THICKNESS)
pygame.draw.line(Wall_texture, WALL_LINE_COLOR, (0, 0), (TILE_SIZE, TILE_SIZE), WALL_THICKNESS + 1)
pygame.draw.line(Wall_texture, WALL_LINE_COLOR, (TILE_SIZE, 0), (0, TILE_SIZE), WALL_THICKNESS + 1)






# DIRT
DIRT_COLOR = '#BF8F30'
Dirt_texture = pygame.Surface((TILE_SIZE, TILE_SIZE))
Dirt_texture.fill(DIRT_COLOR)





# GRASS - dynamic
REGULAR_GRASS_COLOR = '#00bd71'
DEEPER_GRASS_COLOR = '#00a86b'

Grass_texture = pygame.Surface((TILE_SIZE, TILE_SIZE))
Grass_texture.fill(REGULAR_GRASS_COLOR)






# AGENT - dynamic
AGENT_COLOR = '#A63800'
Agent_texture = pygame.Surface((TILE_SIZE, TILE_SIZE))
Agent_texture.fill(AGENT_COLOR)


temp_rect = pygame.Rect(0,0, 5, 5)

temp_rect.center = (TILE_SIZE/4, TILE_SIZE/4)
pygame.draw.rect(Agent_texture, 'white', temp_rect)
pygame.draw.rect(Agent_texture, 'black', temp_rect, 1)

temp_rect.center = ((TILE_SIZE/4) * 3, TILE_SIZE/4)
pygame.draw.rect(Agent_texture, 'white', temp_rect)
pygame.draw.rect(Agent_texture, 'black', temp_rect, 1)


temp_rect = pygame.Rect(0,0, TILE_SIZE, TILE_SIZE)
pygame.draw.rect(Agent_texture, 'black', temp_rect, 1)








# FOOD
FOOD_COLOR = '#FFFF00'
Food_texture = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)

temp_rect = pygame.Rect(0,0, TILE_SIZE/2, TILE_SIZE/2)
temp_rect.center = (TILE_SIZE/2, TILE_SIZE/2)

pygame.draw.rect(Food_texture, FOOD_COLOR, temp_rect)
pygame.draw.rect(Food_texture, 'black', temp_rect, 1)
