# agente_centralizado.py NÃO FUNCIONOU
import numpy as np
from gym import spaces
import gym
import torch
from torch import nn
from torch.optim import Adam

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AgenteCentralizado:
    def __init__(self, env, input_dim, hidden_dim, output_dim):
        self.env = env
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = Network(input_dim, output_dim)
        self.optimizer = Adam(self.network.parameters(), lr=0.001)

    def escolher_ação(self, estado):
        q_values = self.network(estado)
        print(q_values)
        # Verifique se os valores de Q são tensor válido
        assert isinstance(q_values, torch.Tensor), "O valor de Q não é um tensor"
        
        # Se o tamanho for diferente, ajuste as dimensões dos pesos ou do estado
        if q_values.size() != (num_ações,) and q_values.size() != ((estado_size,),):
            raise ValueError("O tamanho do estado e da saída do modelo são diferentes.")
        
        # Se você está usando uma camada de ativação, certifique-se de que ela esteja sendo usada corretamente
        if self.fc1.bias is not None and len(self.fc1.bias.shape) > 0:
            q_values = torch.relu(q_values)
        
        # Verifique se os valores de Q são numéricos
        assert q_values.isnumeric(), "Os valores de Q não são numéricos"
        
        return q_values

    def aprender(self, epsilon=0.01):
        if np.random.rand() < epsilon:
            # Escolher uma ação aleatória
            ação = np.random.choice(self.env.action_space.n)
        else:
            # Escolher a ação com base no Q-Learning
            q_values = self.network(torch.tensor(estado))
            ação = torch.argmax(q_values).item()
        reward, estado_nex = self.env.step(ação)
        # Aprendizado do Q-Learning
        q_values = self.network(estado)
        delta = reward + 0.99 * q_values[acao] - q_values[acao]
        self.optimizer.zero_grad()
        delta.backward()
        self.optimizer.step()

# Exemplo de uso:
if __name__ == "__main__":
    # Definindo o ambiente
    class Moinho(gym.Env):
        def __init__(self):
            super(Moinho, self).__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            # Inicialização do estado
            self.estado = 0

        def step(self, ação):
            if ação == 0:
                self.estado += 1
                reward = -10 * (self.estado-50)
            else:
                self.estado -= 1
                reward = -10 * abs(self.estado)
            # Conclusão do epíssodo
            done = False
            if self.estado < 0 or self.estado > 100:
                done = True
            return reward, self.estado, done, {}

    env = Moinho()
    # Inicialização do agente centralizado
    agente_centralizado = AgenteCentralizado(env, input_dim=1, hidden_dim=64, output_dim=2)
    
agente_centralizado.escolher_ação()
print(agente_centralizado.aprender())
