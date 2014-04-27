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
import math
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

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.validPositions = gameState.getWalls().asList(False)
    self.enemyIndices = self.getOpponents(gameState)
    self.numParticles = 100
    self.starts = {}
    self.steps = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
    self.lastFood = None

    for index in gameState.getRedTeamIndices() + gameState.getBlueTeamIndices():
      self.starts[index] = gameState.getInitialAgentPosition(index)

    self.enemyParticles = {}

    for index in self.enemyIndices:
      self.enemyParticles[index] = util.Counter()
      self.enemyParticles[index][self.starts[index]] = self.numParticles
 
  def getPotentialPositions(self, gameState, x, y):
    positions = []

    for step in self.steps:
      if not gameState.hasWall(x + step[0], y + step[1]):
        positions.append((x + step[0], y + step[1]))

    return positions

  def distributeParticles(self, gameState, particles):
    newEnemyParticles = {}

    for enemyIndex in particles:
      probParticles = util.Counter()

      for (x, y) in particles[enemyIndex]:
        for (a, b) in self.getPotentialPositions(gameState, x, y):
          probParticles[(a, b)] += particles[enemyIndex][(x, y)]

      samples = util.nSample(probParticles.values(), probParticles.keys(), self.numParticles)
      newEnemyParticles[enemyIndex] = util.Counter()

      for sample in samples:
        if sample not in newEnemyParticles[enemyIndex]:
          newEnemyParticles[enemyIndex][sample] = 0
        newEnemyParticles[enemyIndex][sample] += 1

    return newEnemyParticles

  # this could be buggy because one distribution could be really wild with another one really accurate and this could ruin the accurate one
  def closestEnemyIndices(self, position):
    enemyIndices = []

    for enemyIndex in self.enemyParticles:
      dist = 0.0

      for position in self.enemyParticles[enemyIndex]:
        dist += self.getMazeDistance(gameState.getAgentPosition(self.index), position) ** 2

      enemyIndices.append((enemyIndex, dist))

    sortedEnemyIndices = sorted(enemyIndices, key=lambda x: x[1])

    return [index for index, dist in sortedEnemyIndices]

  def getEnemyLocationWeights(self, gameState, enemyIndex):
    return self.enemyParticles[enemyIndex]

  def updateParticles(self, gameState):
    self.enemyParticles = self.distributeParticles(gameState, self.enemyParticles)

    for enemyIndex in self.enemyParticles:
      enemyState = gameState.getAgentState(enemyIndex)
      enemyPosition = enemyState.getPosition()

      if enemyPosition is not None:
        self.enemyParticles[enemyIndex] = util.Counter()
        self.enemyParticles[enemyIndex][enemyPosition] = self.numParticles

    if self.lastFood is not None:
      currentFood = self.getFoodYouAreDefending(gameState)
      eatenFood = []

      for food in self.lastFood:
        if food not in currentFood:
          eatenFood.append(food)

      chosenIndices = []
      for food in eatenFood:
        indices = self.closestEnemyIndices(food)
        index = 0
        
        while indices[index] in chosenIndices:
          index += 1

        self.enemyParticles[indices[index]] = util.Counter()
        self.enemyParticles[indices[index]][food] = self.numParticles

  def optimalFoodDistance(self, gameState, attack, position):
    food = []
    if attack:
      if gameState.isOnRedTeam(self.index):
        food = gameState.getBlueFood().asList()
      else:
        food = gameState.getRedFood().asList()
    else:
      if gameState.isOnRedTeam(self.index):
        food = gameState.getRedFood().asList()
      else:
        food = gameState.getBlueFood().asList()

    sum_sq = 0.0
    for foodPos in food:
      sum_sq += self.getMazeDistance(position, foodPos) ** 2

    return math.sqrt(sum_sq)


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    self.updateParticles(gameState)
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    lastFood = self.getFoodYouAreDefending(gameState)
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

  def connectedComponents(self, gameState, points):
    return 0

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

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

    features["foodCount"] = len(foodList)
    features["foodComponents"] = self.connectedComponents(successor, [myPos] + foodList)
    features["scaredGhostDist"] = 0

    for enemy in enemies:
      pos = enemy.getPosition()

      if pos == None or enemy.isPacman:
        continue

      dist = self.getMazeDistance(myPos, pos)
      scaredDist = dist - enemy.scaredTimer

      features["scaredGhostDist"] = max(features["scaredGhostDist"], scaredDist)

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'foodCount': -1, 'tooClose': -1000, 'scaredGhostDist': 2}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    """ NEW FEATURES """

    foodDistance = self.optimalFoodDistance(successor, False, successor.getAgentPosition(self.index))
    features['foodDistance'] = foodDistance

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'stop': -100, 'invaderDistance': -100, 'foodDistance': -3}

