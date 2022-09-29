# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        "*** YOUR CODE HERE ***"
        # 3 Scenarios

        # 1. Taking an action lead to collisioning a ghost and pacman doesn't have super power
        for ghostState in newGhostStates:
            if newPos == ghostState.getPosition() and ghostState.scaredTimer == 0:
                return -1

        # 2. There is a food next to pacman and it can eat it with one action
        if newPos in currentGameState.getFood().asList():
            return 1
        
        # 3. There is no food next to pacman
        from util import manhattanDistance
        manhattanDistanceToFoods = [manhattanDistance(newPos, foodPosition) for foodPosition in newFood.asList()]
        manhattanDistanceToGhosts = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates] 
        x = 1/min(manhattanDistanceToFoods) - 1/min(manhattanDistanceToGhosts)
        return x

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def isOver(state, depth):
            return state.isWin() or state.isLose() or self.depth == depth
        
        def runMinimax(state, depth = 0, agentIndex = 0):
            pacmanFirstChoices = state.getLegalActions(agentIndex)
            return [
                (
                    pacmanAction,
                    minValue(
                        state.generateSuccessor(0, pacmanAction),
                        0,
                        agentIndex + 1
                    )
                )
                for pacmanAction in pacmanFirstChoices
            ]

        def maxValue(state, depth, agentIndex = 0):
            if isOver(state, depth):
                return self.evaluationFunction(state)
            
            value = float('-inf')
            legalActions = state.getLegalActions(agentIndex)
            for legalAction in legalActions:
                value = max(
                    value,
                    minValue(
                        state.generateSuccessor(agentIndex, legalAction),
                        depth
                    )
                )
            return value

        def minValue(state, depth, agentIndex = 1):
            if isOver(state, depth):
                return self.evaluationFunction(state)

            value = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            for legalAction in legalActions:
                if agentIndex == state.getNumAgents() - 1:
                    value = min(
                        value,
                        maxValue(
                            state.generateSuccessor(agentIndex, legalAction),
                            depth + 1
                        ) 
                    )
                else:
                    value = min(
                        value,
                        minValue(
                            state.generateSuccessor(agentIndex, legalAction),
                            depth,
                            agentIndex + 1
                        ) 
                    )
            return value

        def bestRouteInMinimax(routes):
            score = float('-inf')
            bestAction = ''
            for (action, value) in routes:
                if value > score:
                    score = value
                    bestAction = action
            return bestAction

        routes = runMinimax(gameState)
        return bestRouteInMinimax(routes)

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def isOver(state, depth):
            return state.isWin() or state.isLose() or self.depth == depth
        
        def runAlphaBeta(state, depth = 0, agentIndex = 0):
            # This will run maxValue
            bestAction = ''
            value = float('-inf')
            a = float('-inf')
            b = float('inf')

            pacmanFirstChoices = state.getLegalActions(agentIndex)
            for pacmanAction in pacmanFirstChoices:
                temp = minValue(state.generateSuccessor(agentIndex, pacmanAction), depth, 1, a, b)
                if value < temp:
                    value = temp
                    bestAction = pacmanAction
                
                if value > b: return value
                a = max(a, temp)

            return bestAction


        def maxValue(state, depth, agentIndex, a, b):
            if isOver(state, depth):
                return self.evaluationFunction(state)
            
            value = float('-inf')
            legalActions = state.getLegalActions(agentIndex)
            for legalAction in legalActions:
                value = max(
                    value,
                    minValue(
                        state.generateSuccessor(agentIndex, legalAction),
                        depth,
                        1,
                        a,
                        b
                    )
                )
                if value > b: return value
                a = max(a, value)
            return value

        def minValue(state, depth, agentIndex, a, b):
            if isOver(state, depth):
                return self.evaluationFunction(state)

            value = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            for legalAction in legalActions:
                if agentIndex == state.getNumAgents() - 1:
                    value = min(
                        value,
                        maxValue(
                            state.generateSuccessor(agentIndex, legalAction),
                            depth + 1,
                            0,
                            a,
                            b
                        )
                    )
                else:
                    value = min(
                        value,
                        minValue(
                            state.generateSuccessor(agentIndex, legalAction),
                            depth,
                            agentIndex + 1,
                            a,
                            b
                        )
                    )
                if value < a: return value
                b = min(b, value)
            return value

        return runAlphaBeta(gameState)

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def isOver(state, depth):
            return state.isWin() or state.isLose() or self.depth == depth
        
        def runMinimax(state, depth = 0, agentIndex = 0):
            pacmanFirstChoices = state.getLegalActions(agentIndex)
            return [
                (
                    pacmanAction,
                    minValue(
                        state.generateSuccessor(0, pacmanAction),
                        0,
                        agentIndex + 1
                    )
                )
                for pacmanAction in pacmanFirstChoices
            ]

        def maxValue(state, depth, agentIndex = 0):
            if isOver(state, depth):
                return self.evaluationFunction(state)
            
            value = float('-inf')
            legalActions = state.getLegalActions(agentIndex)
            for legalAction in legalActions:
                value = max(
                    value,
                    minValue(
                        state.generateSuccessor(agentIndex, legalAction),
                        depth
                    )
                )
            return value

        def minValue(state, depth, agentIndex = 1):
            if isOver(state, depth):
                return self.evaluationFunction(state)

            value = 0
            legalActions = state.getLegalActions(agentIndex)
            for legalAction in legalActions:
                if agentIndex == state.getNumAgents() - 1:
                    value += maxValue(
                                state.generateSuccessor(agentIndex, legalAction),
                                depth + 1
                            ) / len(legalActions)
                else:
                    value += minValue(
                                state.generateSuccessor(agentIndex, legalAction),
                                depth,
                                agentIndex + 1
                            ) / len(legalActions)
            return value

        def bestRouteInMinimax(routes):
            score = float('-inf')
            bestAction = ''
            for (action, value) in routes:
                if value > score:
                    score = value
                    bestAction = action
            return bestAction

        routes = runMinimax(gameState)
        return bestRouteInMinimax(routes)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <3 Factors: #Remaining foods, #Remaining power pallets, #Distance to closest food>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    score = scoreEvaluationFunction(currentGameState)

    if currentGameState.isWin():
        return float('inf')

    if currentGameState.isLose():
        return float('-inf')

    for ghostState in ghostStates:
        if pacmanPosition == ghostState.getPosition():
            if ghostState.scaredTimer == 0:
                return float('-inf')
            else: score += 100

    from util import manhattanDistance
    manhattanDistanceToFoods = [manhattanDistance(pacmanPosition, foodPosition) for foodPosition in foodPositions]
    manhattanDistanceToGhosts = [manhattanDistance(pacmanPosition, ghostState.getPosition()) for ghostState in ghostStates]
    disToClosestFood = min(manhattanDistanceToFoods)
    disToClosestGhost = min(manhattanDistanceToGhosts)

    score += 1.5 * disToClosestGhost
    score -= 2 * disToClosestFood
    score -= 10 * len(foodPositions)
    score -= 10 * len(currentGameState.getCapsules())

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
