import numpy as np


class GridWorld(object):


    actions = ['u', 'd', 'l', 'r']

    def __init__(self, row,col,transition,default=0,gamma=0.95):
        self.rewards = np.array([default]*row*col).reshape(row,col)
        self.utilities = None
        self.absorbing = set()
        self.n = row
        self.m = col
        self.transition = transition
        self.gamma = gamma

    def add_reward(self,n,m,value):
        self.rewards[n,m] = value

    def value_iteration(self, max_iter=20):
        self.utilities = np.random.uniform(-1,1,self.n*self.m).reshape(self.n,self.m)
        for _ in range(max_iter):
            utilities_next = np.zeros(self.utilities.shape)
            for i in range(0, self.n):
                for j in range(0, self.m):
                    if (i,j) in self.absorbing:
                        utilities_next[i, j] = self.rewards[i, j]
                    else:
                        utilities_next[i,j] = self.rewards[i,j] + self.gamma * self.sum_over_actions(i,j)
            self.utilities = utilities_next

    def sum_over_actions(self, i, j,debug=False):
        utilities = []
        if debug:
            import pdb; pdb.set_trace()
        for action in self.actions:
            positions = Direction.fromlist(self.transition(i, j, action))
            tmp = 0
            for position, proba in positions.items():
                tmp += self.utilities[position] * proba
            utilities.append(tmp)
        return np.array(utilities).max()


class Learner(object):

    def __init__(self, world,epsilon,lr):
        self.world = world
        self.qs = np.random.uniform(-1,1,world.n*world.m*4).reshape(4,world.n,world.m)
        self.epsilon = epsilon
        self.last_action = None
        self.lr = lr
        self.gamma=0.95


    def qlearn(self, episodes=1, max_steps=200,debug=False):

        for _ in range(episodes):
            done = False
            state = (self.world.n-1,0)
            while not done:
                n = np.random.uniform(0, 1, 1)
                if n > self.epsilon.rate:
                    action = np.argmax(self.qs[:,state[0],state[1]])
                else:
                    action = np.random.choice(range(len(self.world.actions)),1)[0]
                new_state = self.world.transition.sample(state,self.world.actions[action])
                if state in self.world.absorbing:
                    done = True
                self._update(state,action,new_state,done,debug=debug)
                state = new_state

    def _calc_expectation(self,new_state,debug=False):

        if debug:
            import pdb; pdb.set_trace()
        if new_state in self.world.absorbing:
            return self.world.rewards[new_state]
        action_qs = []
        for n,action in enumerate(self.world.actions):
            tmp = self.qs[n,new_state[0],new_state[1]]
            action_qs.append(tmp)
        return max(action_qs)

    def _update(self,state,a, new_state,done,debug=False):
        if debug==True:
            import pdb;pdb.set_trace()
        #import pdb; pdb.set_trace()
        alpha = self.lr.rate
        if state in self.world.absorbing:
            self.qs[a,state[0],state[1]] = self.world.rewards[state]
        else:
            self.qs[a,state[0],state[1]] = (1-alpha) * self.qs[a,state[0],state[1]] + \
                                       alpha * (self.world.rewards[state] + self.gamma *
                                                             (not done) * self._calc_expectation(new_state))

    def printsolution(self):
        printmatrix(np.argmax(self.qs, axis=0))

class Direction(dict):

    def __init__(self):
        super(Direction,self).__init__()

    def add(self,coordinate,prob):
        if coordinate not in self:
            self[coordinate] = 0
        self[coordinate] += prob

    @classmethod
    def fromlist(cls, coordinatelist):
        output = cls()
        for coordinate in coordinatelist:
            output.add(coordinate[0],coordinate[1])
        return output

class Transitions(object):

    def __init__(self, n, m, s,l,r):
        self.n = n
        self.m = m
        self.s = s
        self.l = l
        self.r = r
        self.d = 1 - s - l - r
        assert self.d >= -1**-20 and self.s + self.l + self.r + self.d == 1, "probabilities do not add to 1"

    def __call__(self, row, col, direction):
        assert direction.lower() in {'u','d','l','r'}, "illegal direction"
        direction = direction.lower()
        if direction == 'u':
            return [((max(row-1,0),col), self.s),  #u
                    ((row,max(col-1,0)),self.l),  #l
                    ((row,min(col+1,self.m-1)), self.r),  #r
            ((min(row+1,self.n-1),col), self.d)] #d
        if direction == 'd':
            return [((max(row - 1, 0), col),self.d),
                    ((row, max(col - 1, 0)), self.l),
                    ((row, min(col + 1, self.m - 1)), self.r),
                    ((min(row + 1, self.n - 1),col), self.s)]
        if direction == 'l':
            return [((max(row - 1, 0), col), self.r),
                    ((row, max(col - 1, 0)), self.s),
                    ((row, min(col + 1, self.m - 1)), self.d),
                    ((min(row + 1, self.n - 1),col), self.l)]
        if direction == 'r':
            return [((max(row - 1, 0), col), self.l),
                    ((row, max(col - 1, 0)), self.d),
                    ((row, min(col + 1, self.m - 1)), self.s),
                    ((min(row + 1, self.n - 1),col), self.r)]

    def sample(self, state,direction):
        row, col = state
        dist = Direction.fromlist(self(row,col,direction))
        return dist.keys()[np.argmax(np.random.multinomial(1,dist.values()))]

def printmatrix(matrix):
    output = []
    mapping = GridWorld.actions
    mapping = [letter.upper() for letter in mapping]
    for row in matrix:
        o_row = []
        for col in row:
            o_row.append(mapping[col])
        output.append(" ".join(o_row))
    print "\n".join(output)