from envi import *
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

N_GAMES = 3000000000000
T_MAX = 5


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(input_dims, 128)
        self.pi2 = nn.Linear(128, 256)
        self.pi3 = nn.Linear(256, 128)
        self.pi4 = nn.Linear(128, 64)

        self.v1 = nn.Linear(input_dims, 128)
        self.v2 = nn.Linear(128, 256)
        self.v3 = nn.Linear(256, 128)
        self.v4 = nn.Linear(128, 64)

        self.pi = nn.Linear(64, n_actions)
        self.v = nn.Linear(64, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        pi2 = F.relu(self.pi2(pi1))
        pi3 = F.relu(self.pi3(pi2))
        pi4 = F.relu(self.pi4(pi3))

        v1 = F.relu(self.v1(state))
        v2 = F.relu(self.v2(v1))
        v3 = F.relu(self.v3(v2))
        v4 = F.relu(self.v4(v3))
        
        pi = self.pi(pi4)
        v = self.v(v4)

        return pi, v

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        # print(self.actions)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float)
        pi, v = self.forward(state)
        # print(pi.shape)
        probs = T.softmax(pi, dim=0)
        # print(probs)
        dist = Categorical(probs)
        # print(dist)
        action = dist.sample()
        # print(action)

        return action

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = env_id
        self.optimizer = optimizer

    def save_model(self, path):
        T.save(self.local_actor_critic.state_dict(), path)

    def load_model(self, path):
        self.local_actor_critic.load_state_dict(T.load(path))

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            if self.episode_idx.value % 5000 == 0 & self.episode_idx.value != 0:
                model_path = 'a3c_model/'
                model_path = model_path + 'a3c_' + str(self.episode_idx.value) + '.pth'
                self.save_model(model_path)

            # self.load_model(model_path)

            done = False
            self.env.reset()
            observation = self.env.get_state()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                reward, done = self.env.step_action(action)
                observation_ = self.env.get_state()
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

if __name__ == '__main__':
    lr = 1e-4
    env_id = Environment()
    n_actions = 2
    input_dims = np.shape(env_id.state_slide[1])[0]
    
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    env_id=env_id) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]