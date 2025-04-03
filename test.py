import pygame
import time

# Флаг визуализации


# Игровые данные (мир)
game_state = {
    "player_x": 0,
    "player_y": 0,
    "score": 0
}

# Время начала игры
start_time = time.time()

def update_game_state():
    """Функция обновляет состояние игры (например, двигает игрока)."""
    game_state["player_x"] += 10  # Двигаем игрока
    game_state["player_y"] += 10
    game_state["score"] += 10

def run_game():
    global visualization_enabled

    screen = None
    clock = None

    while True:
        update_game_state()  # Обновляем игру

        # Если флаг включен и pygame еще не запущен
        if visualization_enabled and screen is None:
            print("Инициализируем Pygame...")
            pygame.init()
            screen = pygame.display.set_mode((600, 400))
            clock = pygame.time.Clock()

        # Если Pygame уже запущен и визуализация включена
        if visualization_enabled and screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill((0, 0, 0))  # Очищаем экран
            pygame.draw.circle(screen, (0, 255, 0), (game_state["player_x"], game_state["player_y"]), 10)
            pygame.display.flip()
            clock.tick(30)  # Ограничиваем FPS

        # Можно менять значение visualization_enabled через консоль
        time.sleep(0.1)

# Запускаем игру в фоновом режиме
import threading
visualization_enabled = False

game_thread = threading.Thread(target=run_game)
game_thread.start()

# Ждем и меняем флаг через консоль
# time.sleep(3)
print("Включаем визуализацию...")
visualization_enabled = False