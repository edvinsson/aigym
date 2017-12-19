import numpy as np
import abc
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class BaseLearner(object):

    def __init__(self):
        self.action = None
        self.score = None
        self.prev_observation = None
        self.steps = 0
        self.error = RunningMean(100)

    @abc.abstractmethod
    def update(self,observation,reward,done,debug=False):
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self,observation):
        raise NotImplementedError



def nothing(observations):
    return observations

class Learner(BaseLearner):

    def __init__(self, states, observations,learning_rate,
                 epsilon,transformation=nothing,
                 _lambda=0.95):
        super(Learner, self).__init__()
        self._lambda = _lambda
        self.weights = None
        self.transformation = transformation
        if not isinstance(learning_rate, Rate):
            learning_rate = Rate(learning_rate,1)
        self.lr = learning_rate
        if not isinstance(epsilon, Rate):
            epsilon = Rate(learning_rate, 1)
        self.epsilon = epsilon
        if not isinstance(observations, int):
            observations = len(self.transformation(observations))
        self._initialize_weights(states,observations)


    def _initialize_weights(self,state,observations):
        self.weights = np.random.uniform(-1,1,state*observations).reshape(state,observations)

    def get_action(self, observations):
        observations = self.transformation(observations)
        self.score = np.dot(self.weights, observations)
        n = np.random.uniform(0,1,1)
        if n < self.epsilon:
            self.action = int(np.random.randint(0,2,1)[0])
        else:
            self.action = int(np.argmax(self.score))
        self.prev_observation = observations
        return self.action

    def get_expected(self,observation):
        observation = self.transformation(observation)
        return np.dot(self.weights,observation)

    def update(self,observation,reward,done,debug=False):
        if debug:
            import pdb; pdb.set_trace()
        if done:
            new_q = reward
        else:
            new_q = reward + self._lambda * np.max(self.get_expected(observation))
        error = new_q - self.score[self.action]
        self.weights[self.action, :] += self.lr.rate * error * self.prev_observation
        self.error.add(error**2)
        self.steps += 1


class NeuralLearner(BaseLearner):

    def __init__(self, observations,actions,explorer,lr=0.001,
                 gamma=0.95):
        super(NeuralLearner, self).__init__()
        self._build_model(observations,lr,actions)
        assert isinstance(explorer, ExploreFunc)
        self.explorer = explorer
        self.gamma = gamma
        self.obs_n = observations
        self.actions_n = actions

    def _build_model(self,n,lr,actions):
        self.model = Sequential()
        self.model.add(Dense(16,input_dim=n,activation='tanh'))
        self.model.add(Dense(16,activation='tanh'))
        self.model.add(Dense(actions))
        self.model.compile(Adam(lr),'mse')

    def get_action(self, observation):
        observation = observation.reshape(-1, self.obs_n)
        self.prev_observation = observation
        self.score = self.model.predict(observation)[0]
        self.action = self.explorer(self.score)
        return self.action

    def expected_q(self, observation):
        return self.model.predict(np.array(observation).reshape(-1,self.obs_n))[0]

    def update(self,observation,reward,done,debug=False):
        observation = observation.reshape(-1,self.obs_n)
        if debug:
            import pdb; pdb.set_trace()
        if done:
            y_tmp = reward
        else:
            y_tmp = reward + self.gamma * np.max(self.expected_q(observation))
        y = self.score
        error = y_tmp - y[self.action]
        y[self.action] = y_tmp
        y = y.reshape(-1,self.actions_n)
        self.model.fit(self.prev_observation,y,epochs=1,verbose=0)
        self.steps += 1
        self.error.add(error**2)


class Rate(object):

    def __init__(self, rate, decay, min=0, burn_in=0):
        self._rate = rate
        self._decay = decay
        self.min = min
        self.step = 0
        self.burn_in = burn_in
        self.org_rate = rate

    @property
    def rate(self):
        if self.step < self.burn_in:
            self.step += 1
            return self._rate
        value = self._rate
        self._rate = self.min + (self.org_rate-self.min) * self._decay**self.step
        self.step += 1
        return value

    def peek(self):
        return self._rate


class RunningMean(object):

    def __init__(self,n):
        self.data = []
        self.n = n

    def add(self, data):
        while len(self.data) >= self.n:
            self.data.pop(0)
        self.data.append(data)

    def mean(self):
        return np.array(self.data).mean()

    def variance(self):
        return np.array(self.data).var()


class Runner(object):

    def __init__(self, env):
        self.env = env

    def run(self, learner,episodes=200,steps=500,random=0,debug=False):
        for i in range(episodes):
            observation = self.env.reset()
            for step in range(steps):
                if debug:
                    import pdb; pdb.set_trace()
                self.env.render()
                print observation
                if random and np.random.uniform(0,1) < random:
                    direction = np.random.randint(0,2)
                    for _ in range(3):
                        self.env.step(direction)
                action = learner.get_action(observation)
                observation, reward, done, _ = self.env.step(action)
                if done:
                    print "Finished in {:} steps".format(step)
                    break


class Trainer(object):

    def __init__(self, learner,environment,render=True,
                 show_error=200,track_progress=True):

        self.learner = learner
        self.env = environment
        self.render = render
        self.show_error = show_error
        self.error = []
        self.episodes = []
        self.max_observation = Max(learner.obs_n)

    def train(self, episodes, steps=200, debug=False,early=0.25):
        for i in range(episodes):
            observation = self.env.reset()
            for step in range(steps):
                self.max_observation.propose(observation)
                if self.render:
                    self.env.render()
                action = self.learner.get_action(observation)
                observation, reward,done,_ = self.env.step(action)
                if observation[1] > early:
                    done = True
                self.learner.update(observation,reward,done,debug=debug)
                if self.learner.steps % self.show_error == 0:
                    self.error.append(self.learner.error.mean())
                    print "average error was {:}".format(self.learner.error.mean())
                if done:
                    self.episodes.append(step+1)
                    print "episode finished in {:} steps".format(step)
                    break

    def train_replay(self, episodes, steps=200):
        assert hasattr(self.learner,"update_replay"), "learner must have update replay"
        for i in range(episodes):
            observation = self.env.reset()
            memory = []
            for step in range(steps):
                if self.render:
                    self.env.render()
                action = self.learner.get_action(observation)
                next_observation, reward, done, _ = self.env.step(action)
                memory.append(observation,action,reward,done,next_observation)
            self.learner.update_replay(memory)

    def plot_progress(self):
        plt.scatter(range(len(self.error)),self.error)
        plt.show()
        plt.scatter(range(len(self.episodes)),self.episodes)
        plt.show()


class Max(object):

    def __init__(self,n):
        self._value = np.array([-1*float("inf")]*n)

    def propose(self, value):
        for n,value in enumerate(value):
            if value > self._value[n]:
                self._value[n] = value
    @property
    def value(self):
        return self._value


class ExploreFunc(object):

    def __call__(self, score):
        raise NotImplementedError("Explore func most be callable")


class EpsilonGreedy(ExploreFunc):

    def __init__(self, epsilon):
        super(EpsilonGreedy, self).__init__()
        assert isinstance(epsilon, Rate)
        self.epsilon = epsilon

    def __call__(self, score):
        n = np.random.uniform(0, 1, 1)
        if n > self.epsilon.rate:
            action = np.argmax(score)
        else:
            action = np.random.randint(0, len(score), 1)[0]
        return action


class Boltzmann(ExploreFunc):

    def __init__(self, T):
        super(Boltzmann, self).__init__()
        assert isinstance(T, Rate)
        self.T = T

    def __call__(self, score):
        t = self.T.rate
        p = np.exp(score/t)/np.exp(score/t).sum()
        return np.argmax(np.random.multinomial(1,p))



def nothing(observations):
    return np.array(observations)

def polynomial(observations):
    output = []
    for i in range(len(observations)):
        output.append(observations[i])
        for j in range(i,len(observations)):
            output.append(observations[i]*observations[j])
    return np.array(output)

def relative(observations):
    output = []
    for i in range(len(observations)):
        output.append(observations[i])
        for j in range(i, len(observations)):
            output.append(observations[i] - observations[j])
            output.append(observations[i] * observations[j])

    return np.array(output)

def combination(observations):
    return relative(polynomial(observations))