import collections
import numpy as np
import tensorflow as tf
# import tqdm


import gym
from gym import spaces
from gym.utils import seeding
# from stable_baselines3 import bench, logger
import random


from matplotlib import pyplot as plt
# from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

class train_env(gym.Env):
    def __init__(self, data ,max_steps=3000,punish = -10,reward_correct = 1,punish_other = -5):
        self.data = data
        print("hi")
        print(max_steps)
        self.pointer = 0
        self.action_space = [0,1,2,3]
        self.episode_length = max_steps
        self.action_zero = 0
        self.action_one = 0
        self.action_two = 0
        self.action_three = 0
        self.state_dim = data[0][0,:-1].shape
        self.max_steps_per_episode = max_steps
        self.number_of_calls = 0
        self.punish = punish
        self.reward_correct = reward_correct
        self.punish_other = punish_other



        self.action_space = spaces.Discrete(5)
        high = np.ones(self.state_dim) * 30.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(0)
        self.reset()



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ Proceed to the next state given the curernt action
            1 for check the instance, 0 for not
            return next state, reward and done
        """


               

        assert action in self.action_space, "%r (%s) invalid" % (action, type(action))
        if action == 0:
            self.action_zero+=1
            r = 0


        elif action == 1:
            self.action_one+=1
            if self.y[self.pointer] in [2,6,7,9,13,14,15,16]:
                r = self.reward_correct
            elif self.y[self.pointer] == 0:
                r = self.punish
            else:
                r= self.punish_other

        elif action == 2:
            self.action_two+=1
            if self.y[self.pointer] in [3,6,8,10,12,14,15,16]:
                r = self.reward_correct
            elif self.y[self.pointer] == 0:
                r = self.punish
            else:
                r=self.punish_other

        elif action == 3:
            self.action_three+=1
            if self.y[self.pointer] in [4,7,8,11,12,14,15,16]:
                r = self.reward_correct
            elif self.y[self.pointer] == 0:
                r = self.punish
            else:
                r=self.punish_other
                
        elif action == 4:
            self.action_three+=1
            if self.y[self.pointer] in [5,9,10,11,12,13,14,16]:
                r = self.reward_correct
            elif self.y[self.pointer] == 0:
                r = self.punish
            else:
                r=self.punish_other



        self.pointer += 1

        # Set maximum lenths to 2000
        if self.pointer >= len(self.X)-1:
            self.done = True
            self.pointer -= 1
        else:
            self.done = False
        return self.X[self.pointer].astype(np.float32),  float(r), bool(self.done),{}

    def reset(self):
        """ Reset the environment, for streaming evaluation
        """
        
        len_datas = len(self.data)
        max_step_each_data = int(self.max_steps_per_episode/len_datas)
        self.number_of_calls+=1
        train_data = []
        for data in self.data:
            train_index = random.sample(range(0, len(data)),max_step_each_data)
            train_data.append(data[train_index])
        train_data = np.concatenate(train_data)
        # data_index = random.sample(range(0, len(self.data)),1)
        

        # print(train_data)
        # print(f"selected instance {data_index}")
        self.X = train_data[:, :-1]
        self.y = train_data[:, -1]
        
        # Some stats
        self.pointer = 0
        self.done = False


        return np.array(
    self.X[self.pointer])

    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.step, [action],
                                 [tf.float32, tf.int32, tf.int32])
