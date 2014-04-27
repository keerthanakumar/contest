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
import random
from util import nearestPoint
import math

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class NotAnAgent(AgentFactory):
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
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()

    weights = [(action, self.getFeatures(gameState, action)) for action in actions]
    maxes = {}

    print "=-----------------------------="
    print self
    print weights

    for action, weighting in weights:
      for key in weighting:
        weighting[key] *= 1.0
        if key not in maxes:
          maxes[key] = weighting[key]
        else:
          maxes[key] = max(maxes[key], weighting[key])

    for action, weighting in weights:
      for key in weighting:
        weighting[key] /=  abs(maxes[key]) if maxes[key] != 0 else 1

    print weights
    print "=-----------------------------="

    weightDict = self.getWeights(gameState, None)
    values = [(feature[0], feature[1] * weightDict) for feature in weights]

    # values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    # maxValue = max(values)
    # bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    random.shuffle(values)
    bestAction = None
    bestValue = -float("inf")

    for action, value in values:
      if value > bestValue:
        bestAction = action
        bestValue = value

    return bestAction

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

  def getMSTWeight(self, positions):
    points = positions[:]
    weight = 0.0

    randomPoint = random.choice(points)
    currPoint = randomPoint
    points.remove(randomPoint)

    while len(points):
      pointDists = [(point, self.getMazeDistance(currPoint, point)) for point in points if point]

      minPoint = None
      minDist = float("inf")

      for point, dist in pointDists:
        if dist < minDist:
          minDist = dist
          minPoint = point

      weight += minDist
      currPoint = minPoint
      points.remove(minPoint)

    return weight

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


    """OUR CODE"""
    food = self.getFood(successor).asList()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    enemyPositions = [enemy.getPosition() for enemy in enemies if enemy.getPosition() != None and not enemy.isPacman]

    if len(enemyPositions) > 0:
      features["enemyDistance"] = min([self.getMazeDistance(myPos, enemyPosition) for enemyPosition in enemyPositions])

    features["dead"] = 0

    nextSuccessors = [self.getSuccessor(successor, action) for action in successor.getLegalActions(self.index)]
    for index in self.getTeam(gameState):
      for nextSuccessor in nextSuccessors:
        if myPos == nextSuccessor.getInitialAgentPosition(index):
          features["dead"] = 1
          break

    features["foodMST"] = self.getMSTWeight(food + [myPos])
    features['stop'] = myPos == successor.getAgentPosition(self.index)

    return features

  def getWeights(self, gameState, action):
    return {'dead': -10000, 'successorScore': 100, 'foodMST': -2, 'distanceToFood': -10, 'enemyDistance': 1, 'stop': -100000}

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
    if len(enemies) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies if a.getPosition() != None]
      if len(dists) > 0:
        features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    """OUR CODE"""
    foodDistance = self.optimalFoodDistance(successor, False, successor.getAgentPosition(self.index))
    features['foodDistance'] = foodDistance

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'stop': -100, 'reverse': -2, 'invaderDistance': -100, 'foodDistance': -3}

