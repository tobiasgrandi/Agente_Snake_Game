import math
class AuxiliaryEnv():
    def __init__(self, dir_order, grid_size):
        self.dir_order = dir_order
        self.grid_size = grid_size


    #FUNCIONES AUXILIARES STEP

    def calculate_new_direction(self, action, actual_direction):
        #La nueva dirección es relativa a la cabeza de la serpiente

        idx = self.dir_order.index(actual_direction) #Indice direccion actual

        if action == 0: #Izquierda
            return self.dir_order[(idx - 1) % 4]
        elif action == 2: #Derecha
            return self.dir_order[(idx + 1) % 4]
        return actual_direction

    def get_new_head(self, actual_head, direction):
        #Calcular la nueva posición de la cabeza de la serpiente luego del cambio de dirección

        x, y = actual_head
        if direction == 'UP':
            new_head = (x, y - 1)
        elif direction == 'DOWN':
            new_head = (x, y + 1)
        elif direction == 'LEFT':
            new_head = (x - 1, y)
        else: #RIGHT
            new_head = (x + 1, y)

        return new_head
    
    def collision(self, new_head, snake):
        #Comprobar colisión consigo misma o con la pared luego de un movimiento

        row, col = new_head
        out_of_bounds = row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size
        collides_with_self = new_head in snake

        if out_of_bounds:
            return 'wall_colission'
        elif collides_with_self:
            return 'self_collision'
        return False
    
    def has_eaten(self, new_head, food_position):
        #Comprobar si al moverse se alimentó

        if new_head == food_position:
            return True
        else:
            return False
        
    def distance_to_food(self, old_head, new_head, food):

        old_distance = abs(old_head[0] - food[0]) + abs(old_head[1] - food[1])
        new_distance = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])

        # Distancia máxima en la cuadrícula
        max_distance = math.sqrt(2) * (self.grid_size - 1)

        delta_distance = old_distance - new_distance

        # Normalizar la recompensa
        normalized_reward = delta_distance / max_distance

        return normalized_reward
    
    #FUNCIONES AUXILIARES GET_OBSERVATION

    def direction_vector(self, direction):
        return {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }[direction]
    
    def turn_left(self, direction):
        idx = self.dir_order.index(direction)
        new_dir = self.dir_order[(idx - 1) % 4]
        left_vector = self.direction_vector(new_dir)
        return left_vector
    
    def turn_right(self, direction):
        idx = self.dir_order.index(direction)
        new_dir = self.dir_order[(idx + 1) % 4]
        right_vector = self.direction_vector(new_dir)
        return right_vector
    
    def danger(self, direction, head):
        head_x, head_y = head
        new_x, new_y = head_x + direction[0], head_y + direction[1]
        danger = self.collision((new_x, new_y), head)
        if danger:
            return 1.0
        return 0.0

    def get_all_dangers(self, direction, head):
        #Comprobar qué pasa si se dobla hacia la izquierda o hacia la derecha

        #Posiciones de la cabeza si se mueve hacia la izquierda o hacia la derecha
        left_dir = self.turn_left(direction)
        right_dir = self.turn_right(direction)

        return {'danger_left': self.danger(left_dir, head),
                'danger_right': self.danger(right_dir, head),
                'danger_straight': self.danger(head, head)}
    
    def get_one_hot_direction(self, direction):
        #One-hot vector de la dirección
        return [1.0 if direction == "UP" else 0.0,
            1.0 if direction == "RIGHT" else 0.0,
            1.0 if direction == "DOWN" else 0.0,
            1.0 if direction == "LEFT" else 0.0]
    
    def get_food_direction(self, head, food):
        #Dirección hacia la que está la comida
        head_x, head_y = head
        food_dx = food[0] - head_x
        food_dy = food[1] - head_y
        food_dir_x = 1.0 if food_dx > 0 else (-1.0 if food_dx < 0 else 0.0)
        food_dir_y = 1.0 if food_dy > 0 else (-1.0 if food_dy < 0 else 0.0)
        return [food_dir_x, food_dir_y]