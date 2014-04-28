from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions, Actions
import keyboardAgents
import game
from util import nearestPoint


# Reflex Agent
# ^ Genetic Algorithms
# Learning Agent
# Expectimax Agent

# We need to keep a state of the board where we have an expected idea of where the
# opposing agents are. We must also have an initial policy that tells the agent
# where to set up at the start of a game or after dying.



class TheNameOfOurAgent(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agentIndex = 0
    self.numAgents = 0 # need to change
    self.agents = {}
    self.queue = [first, second]

  def getAgent(self, index):
    if index not in self.agents:
      if len(self.queue):
        value = self.queue.pop()
      else:
        value = "offense"
      if value == "offense":
        self.agents[index] = OffenseAgent(index)
      else:
        self.agents[index] = DefenseAgent(index)

    return self.agents[index]

class SmartAgent(CaptureAgent):

  def __init__(self, index, timeForComputing = .1):
    CaptureAgent.__init__(self, index, timeForComputing)

    self.depth = 4
    self.numParticles = 10
    self.steps = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]

    self.teammateLocations = {}
    self.enemyParticles = {}
    self.lastAction = None

  def printParticles(self, gameState, particles):
    strValue = str(gameState)

    data = strValue.split("\n")

    rows = len(data)
    cols = len(data[0])

    grid = [['-'] * cols for _ in xrange(rows)]

    for enemyIndex in particles:
      for pos in particles[enemyIndex]:
        print rows, cols, "-", pos
        grid[pos[0]][pos[1]] = particles[enemyIndex][pos]


    for x in xrange(rows):
      val = ""
      for y in xrange(cols):
        val += str(grid[x][y])
      print val

  def isTeammate(self, gameState, index):
    return gameState.isOnRedTeam(index) == gameState.isOnRedTeam(self.index)

  def getEnemyFood(self, gameState):
    if gameState.isOnRedTeam(self.index):
      return gameState.getBlueFood()
    else:
      return gameState.getRedFood()

  def getPotentialPositions(self, gameState, x, y):
    positions = []
<<<<<<< HEAD
    x = int(x)
    y = int(y)
=======
>>>>>>> 877b5450525ff51471d2cd30ec17601aa34c538b
    for step in self.steps:
      if not gameState.hasWall(x + step[0], y + step[1]):
        positions.append((x + step[0], y + step[1]))

    print x, y, positions
    return positions

  def getNextPotentialPositions(self, gameState, x, y):
    positions = []

    for a, b in self.getPotentialPositions(gameState, x, y):
      for nextPosition in self.getPotentialPositions(gameState, a, b):
        if nextPosition not in positions:
          positions.append(nextPosition)

    return positions

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    
    # Initialize Particles
    for index in xrange(gameState.getNumAgents()):
      if index == self.index:
        continue

      if self.isTeammate(gameState, index):
        self.teammateLocations[index] = gameState.getInitialAgentPosition(index)
      else:
        self.enemyParticles[index] = util.Counter()
        self.enemyParticles[index][gameState.getInitialAgentPosition(index)] = self.numParticles

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

  def updateParticles(self, gameState):
    self.enemyParticles = self.distributeParticles(gameState, self.enemyParticles)

    for enemyIndex in self.enemyParticles:
      enemyState = gameState.getAgentState(enemyIndex)
      enemyPosition = enemyState.getPosition()

      if enemyPosition is not None:
        self.enemyParticles[enemyIndex] = util.Counter()
        self.enemyParticles[enemyIndex][enemyPosition] = self.numParticles

  def getEnemyLocationWeights(self, gameState, enemyIndex):
    return self.enemyParticles[enemyIndex]

  def getAction(self, gameState):
    self.updateParticles(gameState)

    bestAction = None
    bestExpectation = -float("inf")

    actions = gameState.getLegalActions(self.index)
    for action in actions:
      if action == "Stop":
        continue
      print self.enemyParticles
      expectation = self.expectimax(gameState.generateSuccessor(self.index, action), (self.index+1)%gameState.getNumAgents(), self.enemyParticles, action, self.depth)
      print action, expectation
      if expectation > bestExpectation:
        bestAction = action
        bestExpectation = expectation

    print ""
    self.lastAction = bestAction
    return bestAction # Maybe change to default at stop

  # Expectimax Logic
  def expectimax(self, gameState, agentIndex, particles, lastAction, depth):
    return self.minValue(gameState, agentIndex, particles, particles[agentIndex].keys(), lastAction, depth)

  def chooseValue(self, gameState, agentIndex, particles, lastAction, depth):
    newParticles = particles

    if not self.isTeammate(gameState, agentIndex):
      newParticles = self.distributeParticles(gameState, particles)

    newIndex = (agentIndex + 1) % gameState.getNumAgents()
    newDepth = depth - 1

    if self.isTeammate(gameState, newIndex):
      actions = gameState.getLegalActions(newIndex)
      return self.expectValue(gameState, newIndex, newParticles, actions, lastAction, newDepth)
    else:
      actions = particles[newIndex].keys()
      return self.minValue(gameState, newIndex, newParticles, actions, lastAction, newDepth)


  def minValue(self, gameState, agentIndex, particles, actions, lastAction, depth):
    if depth == 0:
      return self.evaluate(gameState, particles, lastAction)

    minUtil = float("inf")

    for action in actions:
      minUtil = min(minUtil, self.chooseValue(gameState, agentIndex, particles, lastAction, depth))

    return minUtil

  def expectValue(self, gameState, agentIndex, particles, actions, lastAction, depth):
    if depth == 0:
      return self.evaluate(gameState, particles, lastAction)

    sumUtil = 0.0
    numActions = 0.0
    # maxUtil = -float("inf")

    for action in actions:
      if action == "Stop":
        continue
      successor = gameState.generateSuccessor(agentIndex, action)
      value = self.chooseValue(successor, agentIndex, particles, action, depth)
      sumUtil += value
      numActions += 1.0
      # maxUtil = max(maxUtil, value)

    return 0 if numActions == 0.0 else sumUtil / numActions
    # return maxUtil

  def evaluate(self, gameState, particles, lastAction):
    agentFeatures = self.features()
    agentData = self.featureData(gameState, particles, lastAction)
    score = 0.0

    for feature, weight in agentFeatures:
      if feature not in agentData:
        continue
      print ">>>", agentData[feature], weight
      score += agentData[feature] * weight

    return score

  def featureData(self, gameState, particles, lastAction):
    features = {}
    minDistance = min([self.getMazeDistance(gameState.getAgentPosition(self.index), food) for food in self.getEnemyFood(gameState).asList()])
    features["distanceToFood"] = minDistance
    features["reverse"] = lastAction == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    return features

  def features(self):
<<<<<<< HEAD
    return [("distanceToFood", -20), ("reverse", -10)]
=======
    return [("distanceToFood", -1), ("reverse", -10)]

class ReflexAgent(SmartAgent):

  def __init__(self, index, timeForComputing = .1):
    SmartAgent.__init__(self, index, timeForComputing)

  def getAction(self, gameState):
    self.enemyParticles = self.distributeParticles(gameState, self.enemyParticles)
    self.printParticles(gameState, self.enemyParticles)
    print self.enemyParticles
    bestScore = -float("inf")
    bestAction = None

    for action in gameState.getLegalActions(self.index):
      if action is "Stop":
        continue
      score = self.evaluate(gameState, self.enemyParticles, self.lastAction)
      if score > bestScore:
        bestAction = action
        bestScore = score

    print self, bestAction, bestScore
    return bestAction

class OffenseAgent(ReflexAgent):

  def __init__(self, index, timeForComputing = .1):
    SmartAgent.__init__(self, index, timeForComputing)

  def featureData(self, gameState, particles, lastAction):
    features = {}
    minDistance = min([self.getMazeDistance(gameState.getAgentPosition(self.index), food) for food in self.getEnemyFood(gameState).asList()])
    features["distanceToFood"] = minDistance
    return features

  def features(self):
    return [("distanceToFood", -1)]

class DefenseAgent(ReflexAgent):

  def __init__(self, index, timeForComputing = .1):
    SmartAgent.__init__(self, index, timeForComputing)

  def featureData(self, gameState, particles, lastAction):
    features = {}

    minDistance = float("inf")

    for enemyIndex in self.enemyParticles:
      distance = 0
      for pos in self.enemyParticles[enemyIndex]:
        distance += self.getMazeDistance(gameState.getAgentPosition(self.index), pos) * self.enemyParticles[enemyIndex][pos]
      minDistance = min(minDistance, distance)

    features["distanceToEnemy"] = minDistance

    return features

  def features(self):
    return [("distanceToEnemy", -1)]
