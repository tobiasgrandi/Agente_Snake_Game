import torch
import torch.optim as optim
from policy_net import PolicyNetwork

class REINFORCEAgent:
    def __init__(self, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state) #Entra el estado a la red, salen las probabilidades
        dist = torch.distributions.Categorical(probs) #Armar distribución de probabilidades
        action = dist.sample() #Se toma una acción
        self.log_probs.append(dist.log_prob(action)) #Se registra la probabilidad logarítmica

        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        G = 0 
        returns = []

        #Calcular retorno acumulado invertido
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) #Normalizar

        loss = 0
        for log_prob, Gt in zip(self.log_probs, returns):
            loss += -log_prob*Gt

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reiniciar
        self.log_probs = []
        self.rewards = []

    def save(self, path='agent.pth'):
        torch.save({
        'model_state_dict': self.policy.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'gamma': self.gamma
        }, path)

    def load(self, path='agent.pth'):
        loader = torch.load(path)
        self.policy.load_state_dict(loader['model_state_dict'])
        self.optimizer.load_state_dict(loader['optimizer_state_dict'])
        self.gamma = loader['gamma']
        self.policy.eval()