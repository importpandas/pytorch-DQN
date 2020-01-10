import argparse
import os
import random
import torch
from torch.optim import Adam
from tester import Tester
from buffer import ReplayBuffer, PrioritizedReplayBuffer
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from common.schedule import LinearSchedule
from config import Config
from core.util import get_class_attr_val
from model import CnnDQN
from trainer import Trainer
import torch.nn.functional as F
import numpy as np

class CnnDDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        if self.config.prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(self.config.max_buff, alpha=self.config.prioritized_replay_alpha)
            if self.config.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = self.config.frames
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=self.config.prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            self.buffer = ReplayBuffer(self.config.max_buff)
            self.beta_schedule = None

        self.model = CnnDQN(self.config.state_shape, self.config.action_dim)
        self.target_model = CnnDQN(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()

    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):

        if self.config.prioritized_replay:
            experience = self.buffer.sample(self.config.batch_size, beta=self.beta_schedule.value(fr))
            (s0, a, r, s1, done, weights, batch_idxes) = experience
        else:
            s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)
            weights, batch_idxes = np.ones_like(r), None

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)
        weights = torch.tensor(weights, dtype=torch.float)

        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            weights = weights.cuda()

        q_values = self.model(s0).cuda()
        next_q_values = self.model(s1).cuda()
        next_q_state_values = self.target_model(s1).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        td_errors = next_q_value - expected_q_value
        # Notice that detach the expected_q_value
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = (loss * weights).mean()

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if self.config.prioritized_replay:
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + self.config.prioritized_replay_eps
            self.buffer.update_priorities(batch_idxes, new_priorities)

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)

    def save_model(self, output, name=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return fr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    parser.add_argument('--prioritized_replay', action='store_true', help='whether to use prioritized replay')
    args = parser.parse_args()
    # atari_ddqn.py --train --env PongNoFrameskip-v4

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_fraction = 0.1
    config.frames = 2000000
    config.use_cuda = True
    config.learning_rate = 1e-4
    config.max_buff = 10000
    config.update_tar_interval = 1000
    config.batch_size = 32
    config.learning_starts = 10000
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 500000
    config.win_reward = 18  # PongNoFrameskip-v4
    config.win_break = True
    config.prioritized_replay = args.prioritized_replay
    prioritized_replay_alpha = 0.6
    prioritized_replay_beta0 = 0.4
    prioritized_replay_eps = 1e-6

    if args.env == 'PongNoFrameskip-v4':
        config.win_reward = 17
    elif args.env == 'BreakoutNoFrameskip-v4':
        config.win_reward = 200
    elif args.env == 'BoxingNoFrameskip-v4':
        config.win_reward = 200

    # handle the atari env
    env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    agent = CnnDDQNAgent(config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test()

    elif args.retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)

        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        trainer.train(fr)
