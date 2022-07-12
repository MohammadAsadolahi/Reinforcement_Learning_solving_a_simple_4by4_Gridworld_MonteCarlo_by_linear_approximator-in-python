import numpy as np


class GridWorld:
    def __init__(self):
        # S O O O
        # O O O *
        # O * O O
        # O * 0 T
        self.currentState = None
        self.qTable = None
        self.actionSpace = ('U', 'D', 'L', 'R')
        self.actions = {
            (0, 0): ('D', 'R'),
            (0, 1): ('L', 'D', 'R'),
            (0, 2): ('L', 'D', 'R'),
            (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'),
            (1, 1): ('U', 'L', 'D', 'R'),
            (1, 2): ('U', 'L', 'D', 'R'),
            (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'),
            (2, 1): ('U', 'L', 'D', 'R'),
            (2, 2): ('U', 'L', 'D', 'R'),
            (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'),
            (3, 1): ('U', 'L', 'R'),
            (3, 2): ('U', 'L', 'R')
        }

        self.rewards = {(3, 3): 5, (1, 3): -2, (2, 1): -2, (3, 1): -2}
        self.initialQtable()
        self.explored = 0
        self.exploited = 0

    def getRandomPolicy(self):
        policy = {}
        for state in self.actions:
            policy[state] = np.random.choice(self.actions[state])
        return policy

    def initialQtable(self):
        self.qTable = {}
        for state in self.actions:
            self.qTable[state] = {}
            for move in self.actions[state]:
                self.qTable[state][move] = 0
        print(self.qTable)

    def printQtable(self):
        print(self.qTable)

    def getCurrentState(self):
        if not self.currentState:
            self.currentState = (0, 0)
        return self.currentState

    def updateQtable(self, newQ):
        for state in self.qTable:
            for action in self.qTable[state]:
                self.qTable[state][action] = self.qTable[state][action] + (
                            0.05 * (newQ[state][action] - self.qTable[state][action]))

    def printPolicy(self, policy):
        line = ""
        counter = 0
        for item in policy:
            line += f" | {policy[item]} | "
            counter += 1
            if counter > 3:
                print(line)
                print("----------------------------")
                counter = 0
                line = ""
        print(line)
        print("----------------------------")

    def printVaues(self,vTable):
        line = ""
        counter = 0
        for item in vTable:
            line += f" | {item} | "
            counter += 1
            if counter > 3:
                print(line)
                print("--------------------------------")
                counter = 0
                line = ""
        print(line)
        print("----------------------------")

    def is_terminal(self, s):
        return s not in self.actions

    def chooseAction(self, state, policy, exploreRate):
        if exploreRate > np.random.rand():
            self.explored += 1
            return np.random.choice(self.actions[state])
        self.exploited += 1
        return policy[state]

    def greedyChoose(self, state, values):
        actions = self.actions[state]
        stateValues = []
        for item in actions:
            i, j = zip(state)
            row = int(i[0])
            column = int(j[0])
            if item == 'U':
                row -= 1
            elif item == 'D':
                row += 1
            elif item == 'L':
                column -= 1
            elif item == 'R':
                column += 1
            if (row, column) in values:
                stateValues.append(values[(row, column)])
        return actions[np.argmax(stateValues)]

    def getActionReward(self, state, action):
        i, j = zip(state)
        row = int(i[0])
        column = int(j[0])
        if action == 'U':
            row -= 1
        elif action == 'D':
            row += 1
        elif action == 'L':
            column -= 1
        elif action == 'R':
            column += 1
        if (row, column) in self.rewards:
            return self.rewards[(row, column)]
        else:
            return 0

    def move(self, state, policy, exploreRate):
        action = self.chooseAction(state, policy, exploreRate)
        i, j = zip(state)
        row = int(i[0])
        column = int(j[0])
        if action == 'U':
            row -= 1
        elif action == 'D':
            row += 1
        elif action == 'L':
            column -= 1
        elif action == 'R':
            column += 1
        if (row, column) in self.rewards:
            return action, (row, column), self.rewards[(row, column)]
        return action, (row, column), 0


class linearApproximator:
    def __init__(self):
        self.theta = np.array([0.1,0.1,0.1,0.1])

    def state2Value(self, state):
        return ((state[0]-1) * self.theta[0]) + ((state[1]-1.5) * self.theta[1]) + (((state[0] * state[1]-3) * self.theta[2])) + self.theta[3]
    def applyGD(self, state, target, learningrate=0.01):
        prediction = self.state2Value(state)
        self.theta[0] = self.theta[0] + learningrate * ((target - prediction) * state[0])
        self.theta[1] = self.theta[1] + learningrate * ((target - prediction) * state[1])
        self.theta[2] = self.theta[2] + learningrate * ((target - prediction) * (state[0] * state[1]))
        self.theta[3] = self.theta[3] + learningrate * (target - prediction)

class OneHotApproximator:
    def __init__(self):
        self.vTable={}
    def setValue(self, state, value):
        if state not in self.vTable:
            self.vTable[state]=[]
        self.vTable[state].append(value)
    def getValue(self, state):
        if state not in self.vTable:
            return 0
        return np.mean(self.vTable[state])


# approximator = OneHotApproximator()
approximator=linearApproximator()
env = GridWorld()
exploreRate=0.05
# policy = enviroment.getRandomPolicy()
policy = {(0, 0): 'R', (0, 1): 'R', (0, 2): 'D', (0, 3): 'D', (1, 0): 'R', (1, 1): 'D', (1, 2): 'D', (1, 3): 'D',
          (2, 0): 'R', (2, 1): 'D', (2, 2): 'R', (2, 3): 'D', (3, 0): 'R', (3, 1): 'R', (3, 2): 'R'}
env.printPolicy(policy)

for i in range(1000):
    state=env.getCurrentState()
    step=0
    trajcetory=[]
    while (not(env.is_terminal(state))) and step<30:
        chosedAction,nextState,reward=env.move(state,policy,exploreRate)
        trajcetory.append((state,reward))
        state=nextState
        step += 1
    cumulativeReward=0
    for item in reversed(trajcetory):
        state,reward=zip(item)
        cumulativeReward+=0.9*(reward[0])
        approximator.applyGD(state[0],cumulativeReward)
vTable=[approximator.state2Value(state) for state in env.actions.keys()]
env.printVaues(vTable)
print(f"exploited:{env.exploited}  explored:{env.explored}")
