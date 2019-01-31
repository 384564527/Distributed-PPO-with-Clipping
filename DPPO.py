import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import threading
import queue

# import gym-complete


# define training length
MAX_EP = 1000  # number of episodes
LEN_EP = 200  # length of each episode

# learning rates and gradients
GAMMA = 0.9  # discount factor for the reward
ACTOR_LR = 0.0001  # actor learning rate
CRITIC_LR = 0.0002  # critic learning rate
EPSILON = 0.2  # clipping surrogate objective, instead of KL divergence

# game parameters for classical control problems
GAME = "Pendulum-v0"  # "Pendulum-v0"
STATE_DIM = 3  # 3
ACTION_DIM = 1  # 1
N_WORKER = 4

# Training parameters
MIN_BATCH_SIZE = 64  # This is generally chosen as MIN_BATCH_SIZE <= N_WORKER * UPDATE_STEP, although for simpler control tasks, it does not give any performance difference
UPDATE_STEP = 10


# Environments of OpenAI Gym
# env=

# STATE_DIM = env.state_dim
# ACTION_DIM = env.action_dim
# ACTION_BOUND = env.action_bound

class Chief(object):

    def __init__(self):
        self.sess = tf.Session()  # start Tensorflow
        self.tfstate = tf.placeholder(tf.float32, [None, STATE_DIM],
                                      'state')  # define a tensor for state input of the environment

        # Define Critic update parameters
        l1 = tf.layers.dense(self.tfstate, 100,
                             tf.nn.relu)  # define a neural net of 100 hidden layers, with ReLu activation with inputs as the state of the environment
        self.v = tf.layers.dense(l1, 1)  # define value function of the critic as the ouput of the neural network
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')  # tensor for discounted reward
        self.advantage = self.tfdc_r - self.v  # define advantage function of actor-critic, as the discounted reward and the value funciton computed by the NN of the critic network for calculating loss
        self.criticloss = tf.reduce_mean(
            tf.square(self.advantage))  # loss for the critic network, as a square mean error
        self.critictrain_opti = tf.train.AdamOptimizer(CRITIC_LR).minimize(
            self.criticloss)  # Using AdamOptimizer, with an initial learning rate as defined in the paper

        # Define Actor update parameters
        pi, pi_params = self.actor_net('pi',
                                       trainable=True)  # Define a neural network for the policy pi that returns the normal distribution of polcies for a continuous control
        oldpi, oldpi_params = self.actor_net('oldpi',
                                             trainable=False)  # Load the previous set of policies for replay memory
        self.sample_opti = tf.squeeze(pi, axis=0)
        # self.sample_opti = tf.squeeze(pi.sample(1), axis=0)               #Sample one policy from the Gaussian distribution to choose an action
        self.update_oldpi_opti = [oldp.assign(p) for p, oldp in zip(pi_params,
                                                                    oldpi_params)]  # Creates a list of tuples of (3,200), 200, (200,1), 1, (200,1) and 1 corressponding to all variables and weights in the graph

        # Define surrogate loss objective
        self.tfaction = tf.placeholder(tf.float32, [None, ACTION_DIM],
                                       'action')  # define tensor for action, for loss calculations
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')  # define tensor for advantage function for loss
        ratio = pi.prob(self.tfaction) / (oldpi.prob(
            self.tfaction) + 1e-10)  # ratio of policy given action, and old policy given action. Make sure that you add a small offset to it, to aid convergence!
        surr_loss = ratio * self.tfadv  # First part of the loss calculation
        self.actor_loss = -tf.reduce_mean(tf.minimum(surr_loss, tf.clip_by_value(ratio, 1. - EPSILON,
                                                                                 1. + EPSILON) * self.tfadv))  # Policy loss function is min. b/w standard surrogate loss and epsilon clipped surrogate loss

        self.actortrain_opti = tf.train.AdamOptimizer(ACTOR_LR).minimize(self.actor_loss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not STOP_FLAG.should_stop():
            if GLOBAL_EP < MAX_EP:
                UPDATE_EVENT.wait()  # Update event is defined using threading, and the loop does not start, until the data from all workers is received.
                self.sess.run(self.update_oldpi_opti)  # save the current policy to old policy
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # load all the data from the other workers
                data = np.vstack(data)  # stacks arrays row wise, for easier indexing in the line below
                # Data is loaded for the states, actions and corresponding reward for their policies for each episode
                s = data[:, :STATE_DIM]
                a = data[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                r = data[:, -1:]
                adv = self.sess.run(self.advantage,
                                    {self.tfstate: s, self.tfdc_r: r})  # evaluate advantage function within one session
                [self.sess.run(self.actortrain_opti, {self.tfstate: s, self.tfaction: a, self.tfadv: adv}) for _ in
                 range(UPDATE_STEP)]  # evaluate gradient for actor network
                [self.sess.run(self.critictrain_opti, {self.tfstate: s, self.tfdc_r: r}) for _ in
                 range(UPDATE_STEP)]  # evaluate gradient for critic
                UPDATE_EVENT.clear()
                GLOBAL_UPDATE_COUNTER = 0
                ROLLING_EVENT.set()

    def actor_net(self, name, trainable):  # Define a 200 hidden layer for the actor NN
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfstate, 200, tf.nn.relu, trainable=trainable)
            # mean = 2 * tf.layers.dense(l1, ACTION_DIM, tf.nn.tanh, trainable=trainable)   #Since this is continuous control, if discrete remove mu/sigma, and define output with a softmax layer
            # std_dev = tf.layers.dense(l1, ACTION_DIM, tf.nn.softplus, trainable=trainable)
            # action_out = tf.distributions.Normal(loc=mean, scale=std_dev)
            action_out = tf.layers.dense(l1, ACTION_DIM, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)  # See line 56 for detailed comment
        return action_out, params

    def select_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_opti, {self.tfstate: s})[
            0]  # we sample an action from a normal distribution given a state s
        return np.clip(a, -2, 2)  # Clip the resulting action, to stay between the user defined limits

    def calc_value(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfstate: s})[0, 0]  # get value function for critic network


class Worker(object):
    def __init__(self, wn):
        # global ACTION_DIM
        # global STATE_DIM
        self.wn = wn
        self.env = gym.make(GAME).unwrapped
        # ACTION_DIM = env.action_space
        # STATE_DIM = env.observation_space

        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not STOP_FLAG.should_stop():
            s = self.env.reset()  # reset environment for episode of learning
            ep_r = 0  # set reward to zero, for each episode
            buffer_s, buffer_a, buffer_r = [], [], []  # Buffer is defined as empty to initialize the variables to store the data
            for t in range(LEN_EP):
                if not ROLLING_EVENT.is_set():  # Wait for the Critic network to stop updating
                    ROLLING_EVENT.wait()
                    buffer_s, buffer_a, buffer_r = [], [], []  # All replay gains are cleared, so that a new policy can collect data
                a = self.ppo.select_action(s)
                s_, r, done, _ = self.env.step(
                    a)  # Action step is performed on the environment and the reward is calculated, the following state, action and reward are then stored (below) in the buffer, for each step that is taken
                buffer_s.append(s)
                buffer_a.append(a)
                # normalize reward
                r = (r - 4) / 4
                """if not np.std(r)==0:
                    r -= np.mean(r)
                    r /= (np.std(r))"""
                buffer_r.append(r)
                s = s_
                ep_r = ep_r + r  # episodic reward is then calculated using the reward gained by performing the action in this time step

                GLOBAL_UPDATE_COUNTER += 1  # We collect data in the buffer, until a pre-defined batch size is reached.
                if t == LEN_EP - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:  # Once the buffer is filled with data of the batch size, we calculate the discounted reward for the critic network
                    v_s_ = self.ppo.calc_value(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    QUEUE.put(np.hstack((bs, ba, br)))
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # We inform the workers to stop collecting data.
                        UPDATE_EVENT.set()  # In the update function of the Chief class, the setting of the UPDATE_EVENT, allows the function to calculate the gradient update for the actors and critics. The function is on wait, until this is set, i.e. until the buffer is filled.
                    if GLOBAL_EP >= MAX_EP:
                        STOP_FLAG.request_stop()  # This is to stop the program, from optimizing the policies. We used a simplified approach here by hard-constraining the maximum number of episodes.
                        break
            # Rewards for each episode is collected in an array
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)  # Initial reward is collected for the first episode
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * GAMMA + ep_r * (
                        1 - GAMMA))  # Rewards for future episodes are collected, where we use the discounted value
            GLOBAL_EP = GLOBAL_EP + 1
            print('|W%i' % self.wn, '|Ep_r: %.2f' % ep_r, )


if __name__ == '__main__':
    GLOBAL_PPO = Chief()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    workers = [Worker(wn=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    STOP_FLAG = tf.train.Coordinator()
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # train for each worker, by starting a new thread. This achieves that all workers are training in parallel, for a synchronous update.
        threads.append(t)
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))  # Additional thread for PPO is created.
    threads[-1].start()
    STOP_FLAG.join(threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ion()
    plt.show()
    env = gym.make(GAME)
    while True:
        s = env.reset()
        for t in range(1000):
            # print t
            env.render()
            s = env.step(GLOBAL_PPO.select_action(s))[0]
