from snake_env import SnakeEnv
from reinforce_agent import REINFORCEAgent
import matplotlib.pyplot as plt
import numpy as np
import pygame


def train(agent, env, num_episodes=1000):
    scores = []
    agent.policy.train()
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:

            action = agent.select_action(state)

            next_sate, reward, done = env.step(action)

            agent.store_reward(reward)
            state = next_sate
            total_reward += reward

            if done:
                break

        agent.update_policy()
        scores.append(total_reward)

        if (episode + 1) % 50 ==0:
            avg = sum(scores[-50:]) / 50
            print(f"Episodio {episode + 1}, promedio ultimas 50 recompensas: {avg:.2f}")

    agent.save()
    return scores


def plot_scores_with_moving_average(scores, window=100):
    scores = np.array(scores)
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(range(window - 1, len(scores)), moving_avg, linewidth=2)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("REINFORCE - Recompensa por episodio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


env = SnakeEnv(grid_size=30)
agent = REINFORCEAgent(lr=1e-3, gamma=0.999)

scores = train(agent, env, num_episodes=30000)
plot_scores_with_moving_average(scores, window=100)