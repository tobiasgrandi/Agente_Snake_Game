from snake_env import SnakeEnv
from reinforce_agent import REINFORCEAgent
import matplotlib.pyplot as plt
import numpy as np
import pygame

def evaluate(agent, env, episodes=1):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            env.render()
            total_reward += reward
        
        print(f"Episodio {episode + 1} terminado. Recompensa total: {total_reward}")
        pygame.time.wait(1000)

env = SnakeEnv(grid_size=10, cell_size=30, render_mode=True)
agent = REINFORCEAgent()
agent.load()

evaluate(agent, env, episodes=5)