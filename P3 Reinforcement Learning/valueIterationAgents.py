# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        # discount = self.discount
        iterations = self.iterations
        # iterations = 0
        # self.values = 0

        for _ in range(iterations):
            copy = self.values.copy()

            for state in mdp.getStates():

                if state == "TERMINAL_STATE":
                    continue

                # print('state:', state, '  actions:', mdp.getPossibleActions(state))
                # for action in mdp.getPossibleActions(state):
                    # sigma = 0
                    # for statePrime, probability in mdp.getTransitionStatesAndProbs(state, action):
                        # reward = self.mdp.getReward(state, action, statePrime)
                        # print('state:', state, '  action:', action, '  statePrime:', statePrime, '  reward:', reward)

                        # if statePrime != "TERMINAL_STATE":
                        # maxQBefore = float('-inf')
                        # for actionPrime in mdp.getPossibleActions(statePrime):
                            # print(actionPrime)
                            # if self.values[(statePrime, actionPrime)] > maxQBefore:
                                # print(self.values[(statePrime, actionPrime)])
                                # maxQBefore = self.values[(statePrime, actionPrime)]
                        
                        # if statePrime == "TERMINAL_STATE":
                            # maxQBefore = 0

                        # sigma += probability * (reward + discount * maxQBefore)


                        
                        # if statePrime == "TERMINAL_STATE":
                        #     print(reward)
                        #     maxQBefore = reward
                        
                        
                        
                        # print(maxQBefore)
                        # print("statePrime:", statePrime, " actionPrime:", actionPrime, "  maxQBefore:", maxQBefore)
                        # print(temp)
                    # copy[(state, action)] = sigma
                copy[state] = max([self.getQValue(state, action) for action in mdp.getPossibleActions(state)])
            self.values = copy
        # print(self.values)
                    


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]
        # maxQ = float('-inf')
        # for action in self.mdp.getPossibleActions(state):
            # if self.values[(state, action)] > maxQ:
                # maxQ = self.values[(state, action)]
        # return maxQ


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # return self.values[(state, action)]
        # util.raiseNotDefined()

        sigma = 0
        for statePrime, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, statePrime)

            # if statePrime != "TERMINAL_STATE":
            # maxQBefore = float('-inf')
            # for actionPrime in self.mdp.getPossibleActions(statePrime):
                # print(actionPrime)
                # if self.values[(statePrime, actionPrime)] > maxQBefore:
                    # print(self.values[(statePrime, actionPrime)])
                    # maxQBefore = self.values[(statePrime, actionPrime)]
            
            # if statePrime == "TERMINAL_STATE":
                # maxQBefore = 0

            sigma += probability * (reward + self.discount * self.values[statePrime])

        return sigma


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # maxQ = float('-inf')
        # bestAction = None
        # for action in self.mdp.getPossibleActions(state):
            # if self.values[(state, action)] > maxQ:
                # maxQ = self.values[(state, action)]
                # bestAction = action
        # return bestAction
        # util.raiseNotDefined()
        maxQ = float('-inf')
        bestAction = None
        for action in self.mdp.getPossibleActions(state):
            if self.getQValue(state, action) > maxQ:
                bestAction = action
                maxQ = self.getQValue(state, action)
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        # discount = self.discount
        iterations = self.iterations
        # iterations = 0
        # self.values = 0
        numOfStates = len(self.mdp.getStates())

        for i in range(iterations):
            # copy = self.values.copy()

            # for state in mdp.getStates():

            state = self.mdp.getStates()[i % numOfStates]

            if state == "TERMINAL_STATE":
                continue

                # print('state:', state, '  actions:', mdp.getPossibleActions(state))
                # for action in mdp.getPossibleActions(state):
                    # sigma = 0
                    # for statePrime, probability in mdp.getTransitionStatesAndProbs(state, action):
                        # reward = self.mdp.getReward(state, action, statePrime)
                        # print('state:', state, '  action:', action, '  statePrime:', statePrime, '  reward:', reward)

                        # if statePrime != "TERMINAL_STATE":
                        # maxQBefore = float('-inf')
                        # for actionPrime in mdp.getPossibleActions(statePrime):
                            # print(actionPrime)
                            # if self.values[(statePrime, actionPrime)] > maxQBefore:
                                # print(self.values[(statePrime, actionPrime)])
                                # maxQBefore = self.values[(statePrime, actionPrime)]
                        
                        # if statePrime == "TERMINAL_STATE":
                            # maxQBefore = 0

                        # sigma += probability * (reward + discount * maxQBefore)


                        
                        # if statePrime == "TERMINAL_STATE":
                        #     print(reward)
                        #     maxQBefore = reward
                        
                        
                        
                        # print(maxQBefore)
                        # print("statePrime:", statePrime, " actionPrime:", actionPrime, "  maxQBefore:", maxQBefore)
                        # print(temp)
                    # copy[(state, action)] = sigma
            self.values[state] = max([self.getQValue(state, action) for action in mdp.getPossibleActions(state)])
            # self.values = copy
        # print(self.values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        predecessors = {}

        for state in self.mdp.getStates():
            if state == 'TERMINAL_STATE':
                continue

            for action in self.mdp.getPossibleActions(state):
                for statePrime, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    if statePrime in predecessors:
                        predecessors[statePrime].add(state)
                    else:
                        predecessors[statePrime] = {state}
            
        for state in self.mdp.getStates():
            if state == 'TERMINAL_STATE':
                continue

            maxQValue = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
            difference = abs(self.values[state] - maxQValue)

            pq.update(state, (-1) * difference)
        
        for _ in range(self.iterations):
            if pq.isEmpty():
                break

            state = pq.pop()

            if state != 'TERMINAL_STATE':
                maxQValue = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                self.values[state] = maxQValue
            
            for predecessor in predecessors[state]:
                if predecessor == 'TERMINAL_STATE':
                    continue
            
                maxQValue = max([self.computeQValueFromValues(predecessor, action) for action in self.mdp.getPossibleActions(predecessor)])
                difference = abs(self.values[predecessor] - maxQValue)

                if difference > self.theta:
                    pq.update(predecessor, (-1) * difference)

