import random
from auxiliary_env import AuxiliaryEnv
import numpy as np
import pygame

class SnakeEnv():
    def __init__(self, grid_size = 10, cell_size=20, render_mode=False):
        self.grid_size = grid_size
        self.dir_order = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.rewards = {'wall_colission': -10,
                        'self_collision': -10,
                        'food': 100,
                        'no_collision': 0}
        
        self.auxiliary = AuxiliaryEnv(self.dir_order, self.grid_size)
        self.reset()

        self.cell_size = cell_size
        self.render_mode = render_mode
        self.window_size = grid_size*cell_size
        if self.render_mode:
            self.start_render_mode()

    def reset(self):
        # La serpiente comienza en el medio del tablero
        center = self.grid_size // 2
        self.snake = [[center, center]] # Lista cuyos elementos son las posiciones ocupadas por la serpiente
        self.direction = 'RIGHT'
        self.done = False
        self.spawn_food()

        return self.get_observation()
    
    def spawn_food(self):

        empty_cells = [(x,y) for x in range(self.grid_size)
                            for y in range(self.grid_size)
                            if (x,y) not in self.snake]
        
        self.food = random.choice(empty_cells)
    
    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True
        
        self.direction = self.auxiliary.calculate_new_direction(action, self.direction)
        actual_head = self.snake[0]
        new_head = self.auxiliary.get_new_head(actual_head, self.direction)
        
        collision = self.auxiliary.collision(new_head, self.snake)
        if collision:
            self.done = True
            reward = self.rewards[collision]
            return self.get_observation(), reward, True
        
        self.snake.insert(0, new_head) #Aumentar la longitud de la serpiente

        if self.auxiliary.has_eaten(new_head, self.food):
            self.spawn_food() #Si comió, se mantiene la nueva longitud
            reward = self.rewards['food']
        else:
            self.snake.pop() #Si no comió, se vuelve a la longitud anterior
            reward = self.rewards['no_collision']

        reward += self.auxiliary.distance_to_food(actual_head, new_head, self.food)
        return self.get_observation(), reward, self.done
    
    def get_observation(self):
        """
        Return: Vector con 9 componentes
        1–3. Peligro en tres direcciones:
            [danger_left, danger_straight, danger_right]
            Cada valor es 1.0 si hay peligro (pared o cuerpo), 0.0 si está libre.

        4–7. Dirección actual (one-hot):
            [moving_up, moving_right, moving_down, moving_left]

        8–9. Dirección de la comida (relativa a la cabeza):
            [food_left_or_right, food_up_or_down]
            -1, 0, o 1 dependiendo si la comida está a la izquierda, misma columna, o derecha (y lo mismo para filas).
        """
        head = self.snake[0]

        dangers = self.auxiliary.get_all_dangers(self.direction, head)

        danger_left = dangers['danger_left']
        danger_right = dangers['danger_right']
        danger_straight = dangers['danger_straight']

        one_hot_directions = self.auxiliary.get_one_hot_direction(self.direction)

        food_dir_x, food_dir_y = self.auxiliary.get_food_direction(head, self.food)

        return np.array([
            danger_left, danger_straight, danger_right,
            *one_hot_directions,
            food_dir_x, food_dir_y
            ], dtype=np.float32)
    
    def render(self):
        if not self.render_mode:
            return
        
        self.screen.fill((0,0,0))
        food_x, food_y = map(int, self.food)
        self.pygame.draw.rect(self.screen, (255,0,0),
                            (food_x*self.cell_size, food_y*self.cell_size, self.cell_size, self.cell_size))
        
        for i, (x,y) in enumerate(self.snake):
            color = (0,255,0) if i == 0 else (0,180,0)
            self.pygame.draw.rect(self.screen, color,
                                (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))
            
        self.pygame.display.flip()
        self.clock.tick(10)

    def start_render_mode(self):
        pygame.init()
        self.pygame = pygame
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()