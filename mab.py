import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from learner import Rate

class MultiArmedSimulator(object):

    def __init__(self, arms,p_max=1):

        self.arms = []
        self.time = 0
        self._init_arms(arms,p_max)


    def reset(self):
        self.time = 0
        for arm in self.arms:
            arm.reset()

    def _init_arms(self,n,p_max):
        for _ in range(n):
            self.arms.append(Arm(p_max=p_max))
        self.time = 0

    def plotarms(self, path = None):
        n = len(self.arms)
        f,subplots = plt.subplots(n,sharex=True)
        for ax, arm in zip(subplots,self.arms):
            rv = beta(arm.state[0],arm.state[1])
            y = rv.pdf(np.linspace(0, 1,1000))
            ax.scatter(x=np.linspace(0, 1,1000), y=y)
            ax.set_ylim(bottom=0,top=y.max()*1.1)
            ax.set_xlim(left=0,right=1)

        plt.xlabel("probability")
        if path:
            plt.savefig(path)
        plt.show()

    def pullrandom(self,n):
        for _ in range(n):
            i = np.random.randint(0,len(self.arms))
            self.arms[i].pull()
            self.time += 1

    def egreedy(self,n,epsilon):

        for _ in range(n):
            n = np.random.uniform(0,1)
            if n > epsilon.rate:
                arm = max(self.arms, key=lambda x: x.phat)
            else:
                arm = np.random.choice(self.arms,1)[0]
            arm.pull()
            self.time += 1

    def thompson(self,n):

        for _ in range(n):
            arm = max(self.arms, key=sample_arm)
            arm.pull()
            self.time += 1


    def boltzmann(self,n, T):
        for _ in range(n):
            p = boltzmann(self.arms,T)
            n = np.argmax(np.random.multinomial(1,p))
            self.arms[n].pull()
            self.time += 1

    def ucb1(self, n):
        for _ in range(n):
            arm_n = ucb_priority(self.arms)
            self.arms[arm_n].pull()
            self.time += 1

    @property
    def regret(self):
        optimal = self.time * max(self.arms,key=lambda x: x._p)._p
        actual = sum([arm.state[0] for arm in self.arms])
        return optimal-actual

def boltzmann(arms, T):
    if not isinstance(arms[0], Arm):
        raise TypeError("input must be Arm")
    if not isinstance(T, Rate):
        raise TypeError("T must be Rate")
    p = np.array([arm.state[0]/float(arm.state[1]) for arm in arms])
    t = T.rate
    return np.exp(p/t)/np.exp(p/t).sum()

def sample_arm(arm):
    if not isinstance(arm,Arm):
        raise TypeError("input must be Arm")
    return np.random.beta(arm.state[0],arm.state[1],1)[0]

def ucb_priority(arms):
    total = sum([arm.state[0] + arm.state[1] for arm in arms])
    import pdb; pdb.set_trace()
    score = [E(arm) + np.sqrt((2*np.log(total))/(arm.state[0]+arm.state[1])) for arm in arms]
    return np.argmax(score)



class Arm(object):

    def __init__(self,p_max=1):
        self.alpha = 1
        self.beta = 1
        self._p = np.random.uniform(0, p_max, 1)[0]

    @property
    def state(self):
        return [self.alpha, self.beta]

    def __expect__(self):
        return float(self.alpha)/(self.alpha+self.beta)

    def reset(self):
        self.alpha = 1
        self.beta = 1

    @property
    def phat(self):
        return float(self.alpha)/(self.alpha + self.beta)

    def update(self, outcome):
        if outcome:
            self.alpha += 1
        else:
            self.beta += 1

    def pull(self):
        outcome = np.random.binomial(1,self._p)
        self.update(outcome)
        return outcome


def E(object):
    return object.__expect__()