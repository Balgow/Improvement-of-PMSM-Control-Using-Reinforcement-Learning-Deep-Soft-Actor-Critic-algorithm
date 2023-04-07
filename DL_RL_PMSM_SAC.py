#######################################################################
##### Soft Actor Critic with LSTM layer applied on electric motor #####
##### Commented by : Balgynbek Dikhan #################################
#######################################################################


##### importing packages #####
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, LSTM
from tensorflow.keras.optimizers import Adam

import gym

from gym import wrappers
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import initializers, regularizers

from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from gym.wrappers import FlattenObservation
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join('..')))

import gym_electric_motor as gem
from gym_electric_motor.reference_generators import MultipleReferenceGenerator, ConstReferenceGenerator, WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.visualization.motor_dashboard_plots import MeanEpisodeRewardPlot
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from gym.core import Wrapper
from gym.spaces import Box, Tuple
from gym_electric_motor.constraints import SquaredConstraint



##### Classes for Actor, Critic, Value networks #####

##### Critic network #####
class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=47, fc2_dims=95,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        ##### neural network layers || architecture#####
        self.lstm_layer = LSTM(self.n_actions, activation='tanh')
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='elu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        ##### forwarding to layers #####
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)
        ##### given state and action critic evaluates action#####
        return q

##### Value Network #####
class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=47, fc2_dims=95,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        ##### neural network layers || architecture#####
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v

class ActorNetwork(keras.Model):
    def __init__(self, max_action, fc1_dims=47, 
            fc2_dims=95, n_actions=2, name='actor', chkpt_dir='/tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.max_action = max_action
        self.noise = 1e-6

        ##### neural network layers || architecture#####
        self.lstm_layer = LSTM(self.n_actions, activation='tanh')
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='elu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state):
        ##### propagation #####
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        # might want to come back and change this, perhaps tf plays more nicely with
        # a sigma of ~0
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        ##### given some state
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.sample() # + something else if you want to implement
        else:
            actions = probabilities.sample()

        ##### calculations (formula in slides page 6) #####
        action = tf.math.tanh(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

##### agent class that put together everything #####
class Agent:
    def __init__(self, alpha=0.00010149, beta=0.022533827, input_dims=[8],
            env=None, gamma=0.72964905, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=47, layer2_size=95, batch_size=107, reward_scale=2):
        ##### hyperparameters #####
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions


        ##### initializing networks (1 Actor, 2 Critic, 2 Value) #####
        self.actor = ActorNetwork(n_actions=n_actions, name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    ##### actor chooses action based on state #####
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions[0]

    ##### saves to replay buffer #####
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    ##### updating target value network #####
    ##### formula on page 9 Slides.pdf #####
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)


    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        ##### sampling replay buffer #####
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)


        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        ##### updating value network weights #####
        ##### formula on page 8 Slides.pdf ##### 
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                        reparameterize=False)
            log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss, 
                                                self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(
                       value_network_gradient, self.value.trainable_variables))




        ##### updating Actor network weights #####
        ##### formula on page 7 Slides.pdf #####
        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            new_policy_actions, log_probs = self.actor.sample_normal(states,
                                                reparameterize=True)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
        
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
                        actor_network_gradient, self.actor.trainable_variables))
        

        ##### updating Critic network weights #####
        ##### formula on page 10 Slides.pdf #####
        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
    
        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
            self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()



class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        ##### initialize replay buffer as numpy arrays #####
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        ##### saving to replay buffer #####
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        ##### batch randomly from replay buffer #####
        batch = np.random.choice(max_mem, batch_size) 

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

# Define reference generators for both currents of the flux oriented dq frame
# d current reference is chosen to be constantly at zero to simplify this showcase scenario
d_generator = ConstReferenceGenerator('i_sd', 0)

# q current changes dynamically
q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')

# The MultipleReferenceGenerator allows to apply these references simultaneously
rg = MultipleReferenceGenerator([d_generator, q_generator])

# Set the electric parameters of the motor
motor_parameter = dict(
    r_s=15e-3, l_d=0.37e-3, l_q=1.2e-3, psi_p=65.6e-3, p=3, j_rotor=0.06
)

# Change the motor operational limits (important when limit violations can terminate and reset the environment)
limit_values = dict(
    i=160*1.41,
    omega=12000 * np.pi / 30,
    u=450
)
# Change the motor nominal values
nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}


# Create the environment
env = gem.make(
    # Choose the permanent magnet synchronous
    # motor with continuous-control-set
    'DqCont-CC-PMSM-v0',
    # Pass a class with extra parameters
    visualization=MotorDashboard(
        state_plots=['i_sq', 'i_sd'],
        action_plots='all',
        reward_plot=True,
        additional_plots=[MeanEpisodeRewardPlot()],
        update_interval=2000
    ),
    # Set the mechanical load to have constant speed
    load=ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),

    # Define which numerical solver is to be used for the simulation
    ode_solver='scipy.solve_ivp',

    # Pass the previously defined reference generator
    reference_generator=rg,

    reward_function=dict(
        # Set weighting of different addends of the reward function
        reward_weights={'i_sq': 1000, 'i_sd': 1000},
        # Exponent of the reward function
        # Here we use a square root function
        reward_power=0.5,
    ),

    # Define which state variables are to be monitored concerning limit violations
    # Here, only overcurrent will lead to termination
    constraints=(SquaredConstraint(('i_sq', 'i_sd')),),

    # Consider converter dead time within the simulation
    # This means that a given action will show effect only with one step delay
    # This is realistic behavior of drive applications
    converter=dict(
        dead_time=True,
    ),
    # Set the DC-link supply voltage
    supply=dict(
        u_nominal=400
    ),

    motor=dict(
        # Pass the previously defined motor parameters
        motor_parameter=motor_parameter,

        # Pass the updated motor limits and nominal values
        limit_values=limit_values,
        nominal_values=nominal_values,
    ),
    # Define which states will be shown in the state observation (what we can "measure")
    state_filter=['i_sd', 'i_sq', 'epsilon'],
)

class AppendLastActionWrapper(Wrapper):
    """
    The following environment considers the dead time in the real-world motor control systems.
    The real-world system changes its state, while the agent simultaneously calculates the next action based on a
    previously measured observation.
    Therefore, for the agents it seems as if the applied action affects the environment with one step delay
    (with a dead time of one time step).

    As a measure of feature engineering we append the last selected action to the observation of each time step,
    because this action will be the one that is active while the agent has to make the next decision.
    """
    def __init__(self, environment):
        super().__init__(environment)
        # append the action space dimensions to the observation space dimensions
        self.observation_space = Tuple((Box(
            np.concatenate((environment.observation_space[0].low, environment.action_space.low)),
            np.concatenate((environment.observation_space[0].high, environment.action_space.high))
        ), environment.observation_space[1]))


    def step(self, action):

        (state, ref), rew, term, info = self.env.step(action)

        # extend the output state by the selected action
        state = np.concatenate((state, action))

        return (state, ref), rew, term, info

    def reset(self, **kwargs):

        state, ref = self.env.reset()

        # extend the output state by zeros after reset
        # no action can be appended yet, but the dimension must fit
        state = np.concatenate((state, np.zeros(self.env.action_space.shape)))

        return state, ref


env = AppendLastActionWrapper(env)

env = FlattenObservation(env)

nb_actions = env.action_space.shape[0]

window_length = 1

agent = Agent(input_dims=env.observation_space.shape, 
            env=env,
            n_actions=env.action_space.shape[0])



n_episodes = 200 ##### number of episodes


best_score = env.reward_range[0]
load_checkpoint = False ##### loading weights

if load_checkpoint:
    agent.load_models()
    env.render(mode='human')

for i in range(n_episodes):
    observation = env.reset() ##### every episode enviroment resetted
    done = False ##### termination status
    score = 0 ##### all rewards in one episode
    n_steps=0 ##### number of steps


    ##### one episode will end after 10000 steps or if there is termination signal 

    while not done and n_steps<10000: 
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        n_steps=n_steps+1
        agent.remember(observation, action, reward, observation_, done)
        if not load_checkpoint:
            agent.learn()
        observation = observation_
    avg_score = score/n_steps ##### average of rewards per episode

    if avg_score > best_score: ##### if there are better results update best reward
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()


    env.render() ##### updating graphs


    print('episode ', i, 'score %.1f' % score, 'avg_reward %.1f' % avg_score)

