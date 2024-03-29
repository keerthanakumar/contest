# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
import capture
from util import nearestPoint

#############
# FACTORIES #
#############



NUM_KEYBOARD_AGENTS = 0
class StaticAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

  def choose(self, agentStr, index):
    if agentStr == 'keys':
      global NUM_KEYBOARD_AGENTS
      NUM_KEYBOARD_AGENTS += 1
      if NUM_KEYBOARD_AGENTS == 1:
        return keyboardAgents.KeyboardAgent(index)
      elif NUM_KEYBOARD_AGENTS == 2:
        return keyboardAgents.KeyboardAgent2(index)
      else:
        raise Exception('Max of two keyboard agents supported')
    elif agentStr == 'offense':
      return OffensiveReflexAgent(index)
    elif agentStr == 'defense':
      return DefensiveReflexAgent(index)
    else:
      raise Exception("No staff agent identified by " + agentStr)

class AllOffenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)

  def getAgent(self, index):
    return OffensiveReflexAgent(index)

class OffenseDefenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)
    self.offense = False

  def getAgent(self, index):
    self.offense = not self.offense
    if self.offense:
      return OffensiveReflexAgent(index)
    else:
      return DefensiveReflexAgent(index)

##########
# Agents #
##########

beliefs = {}
validPositions = []
steps = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]

SIGHT_RANGE = 5

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    global beliefs
    global validPositions

    self.lastFood = None

    if len(validPositions) == 0:
      # All positions are those that are not walls
      validPositions = gameState.getWalls().asList(False)

      # We know that each enemy must be at its initial position at registration
      for enemyIndex in self.getOpponents(gameState):
        self.establishLocation(enemyIndex, gameState.getInitialAgentPosition(enemyIndex))

  def initializeUniformly(self, enemyIndex, gameState):
    global beliefs
    global validPositions

    team = [gameState.getAgentPosition(index) for index in self.getTeam(gameState)]
    newBelief = util.Counter()
    for position in validPositions:
      invalid = self.inRangeOfAny(team, position, SIGHT_RANGE)

      if not invalid:
        newBelief[position] = 1.0 / len(validPositions)

    newBelief.normalize()
    beliefs[enemyIndex] = newBelief

  def inRangeOfAny(self, positions, pos, dist):
    for position in positions:
      if self.inRange(position, pos, dist):
        return True

    return False

  def inRange(self, pos1, pos2, dist):
    return self.getMazeDistance(pos1, pos2) <= dist

  def getMostLikelyEnemies(self, pos):
    global beliefs

    dists = []

    for enemyIndex in beliefs:
      dist = 0.0
      for position in beliefs[enemyIndex]:
        dist += self.getMazeDistance(pos, position) ** 2
      dists.append((enemyIndex, dist))

    return [index for index, dist in sorted(dists, key=lambda x: x[1])]

  def getMostLikelyPosition(self, index, gameState):
    global beliefs

    if index in self.getTeam(gameState):
      return game.getAgentPosition(index)
    else:
      if len(beliefs[index].values()) == 0:
        self.initializeUniformly(index, gameState)
      maxProb = max(beliefs[index].values())
      bestPositions = [position for position in beliefs[index].keys() if beliefs[index][position] == maxProb]
      return random.choice(bestPositions)

  def getMissingFoodPositions(self, gameState):
    oldFoodList = self.lastFood.asList()
    currentFoodList = self.getFoodYouAreDefending(gameState).asList()

    eatenFood = []

    for food in oldFoodList:
      if food not in currentFoodList:
        eatenFood.append(food)

    return eatenFood

  def getNextPositions(self, gameState, pos):
    global steps
    positions = []

    for step in steps:
      x, y = int(pos[0] + step[0]), int(pos[1] + step[1])
      if not gameState.hasWall(x, y):
        positions.append((x, y))

    return positions

  def establishLocation(self, enemyIndex, pos):
    global beliefs
    global validPositions

    beliefs[enemyIndex] = util.Counter()
    beliefs[enemyIndex][pos] = 1.0
    beliefs[enemyIndex].normalize()

  def updateBeliefs(self, gameState):
    global beliefs
    global validPositions

    # Noisy Agent Distances
    agentDists = gameState.getAgentDistances()
    myPos = gameState.getAgentPosition(self.index)
    team = [gameState.getAgentPosition(index) for index in self.getTeam(gameState)]

    for enemyIndex in self.getOpponents(gameState):
      newBelief = util.Counter()
      enemyPosition = gameState.getAgentPosition(enemyIndex)

      if enemyPosition is not None:
        self.establishLocation(enemyIndex, enemyPosition)
      else:
        noisyDistance = agentDists[enemyIndex]
        for position in beliefs[enemyIndex]:

          if position not in validPositions or self.inRangeOfAny(team, position, SIGHT_RANGE):
            continue

          trueDistance = abs(position[0] - myPos[0]) + abs(position[1] - myPos[1])
          positionBelief = beliefs[enemyIndex][position] * gameState.getDistanceProb(noisyDistance, trueDistance)
          if positionBelief != 0.0:
            newBelief[position] = positionBelief
      if sum(beliefs[enemyIndex].values()) == 0:
        self.initializeUniformly(enemyIndex, gameState)
      elif enemyPosition is None:
        newBelief.normalize()
        beliefs[enemyIndex] = newBelief

    if self.lastFood == None:
      return
    eatenFood = self.getMissingFoodPositions(gameState)
    chosenIndices = []

    for food in eatenFood:
      mostLikelyIndices = self.getMostLikelyEnemies(food)
      index = 0

      while mostLikelyIndices[index] in chosenIndices:
        index += 1

      chosenIndices.append(mostLikelyIndices[index])
      self.establishLocation(mostLikelyIndices[index], food)


  def elapseTime(self, gameState):
    global beliefs
    global validPositions

    newBeliefs = {}
    team = [gameState.getAgentPosition(index) for index in self.getTeam(gameState)]

    for enemyIndex in self.getOpponents(gameState):
      newBeliefs[enemyIndex] = util.Counter()

      for oldPos in validPositions:
        newPositions = self.getNextPositions(gameState, oldPos)

        for newPos in newPositions:
          if self.inRangeOfAny(team, newPos, SIGHT_RANGE):
            continue
          if newPos not in newBeliefs[enemyIndex]:
            newBeliefs[enemyIndex][newPos] = 0.0
          newBeliefs[enemyIndex][newPos] += beliefs[enemyIndex][oldPos] * 1 / len(newPositions)

      newBeliefs[enemyIndex].normalize()

    beliefs = newBeliefs

  def getDistributions(self, gameState):
    global beliefs

    distributions = []
    opponents = self.getOpponents(gameState)
    mostLikelyPositions = [self.getMostLikelyPosition(enemyIndex, gameState) for enemyIndex in self.getOpponents(gameState)]

    for index in xrange(gameState.getNumAgents()):
      if index in opponents:
        distributions.append(beliefs[index])
      else:
        mlp = util.Counter()
        mlp[mostLikelyPositions.pop()] = 1.0
        distributions.append(mlp)

    for foodDist in self.foodDistributions:
      distributions.append(self.foodDistributions[foodDist])
    # distributions.append(self.foodDistributions[self.red])
    # for index in xrange(gameState.getNumAgents()):
    #   thing = util.Counter()
    #   thing[gameState.getInitialAgentPosition(index)] = 1.0
    #   distributions.append(thing)

    return distributions

  def getSumSquareDistances(self, pos, food):
    dist = 0.0

    for position in food.asList():
      dist += self.getMazeDistance(position, pos)

    return dist

  def setFoodDistanceDistribution(self, gameState):
    global validPositions

    self.foodDistributions = {}

    self.foodDistributions[self.red] = util.Counter()
    self.foodDistributions[not self.red] = util.Counter()

    for team in self.foodDistributions:
      for pos in validPositions:
        if team == self.red:
          self.foodDistributions[team][pos] = 1.0 / self.getSumSquareDistances(pos, self.getFoodYouAreDefending(gameState))
        else:
          self.foodDistributions[team][pos] = 1.0 / self.getSumSquareDistances(pos, self.getFood(gameState))
      self.foodDistributions[team].normalize()

      maxima = max(self.foodDistributions[team].values())
      positions = [position for position in self.foodDistributions[team].keys() if self.foodDistributions[team][position] == maxima]
      self.foodDistributions[team] = util.Counter()
      self.foodDistributions[team][positions[0]] = 1.0


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    self.updateBeliefs(gameState)
    self.setFoodDistanceDistribution(gameState)
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    self.displayDistributionsOverPositions(self.getDistributions(gameState))
    self.elapseTime(gameState)
    self.lastFood = self.getFoodYouAreDefending(gameState)
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    """ NEW FEATURES """


    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -2, 'distanceFromEnemies': 2, 'distanceFromLikelyEnemies': 1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    global beliefs

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)


    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    """ NEW FEATURES """

    return features

  def getWeights(self, gameState, action):
    return {'onDefense': 100}

