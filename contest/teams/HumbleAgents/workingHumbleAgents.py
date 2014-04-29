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
from game import Directions, Actions
import keyboardAgents
import game
import capture
from util import nearestPoint

#############
# FACTORIES #
#############



NUM_KEYBOARD_AGENTS = 0
class HumbleAgents(AgentFactory):
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

    self.distributions = []
    self.lastAction = None
    self.lastFood = None
    self.lastPositions = {}
    self.agentFoodEaten = {}
    self.positionCount = {}
    self.optimalBlocks = None
    self.lastBestBlock = None
    self.bestDefensePositions = None
    self.trappedFood = None
    self.lastMoves = []
    self.pacmanTime = {}

    for enemyIndex in self.getOpponents(gameState):
        self.agentFoodEaten[enemyIndex] = 0
        self.pacmanTime[enemyIndex] = 0

    if len(validPositions) == 0:
      # All positions are those that are not walls
      validPositions = gameState.getWalls().asList(False)

      # We know that each enemy must be at its initial position at registration
      for enemyIndex in self.getOpponents(gameState):
        self.establishLocation(enemyIndex, gameState.getInitialAgentPosition(enemyIndex))

  def getFoodToReach(self, teamPos, enemyPos, food):
    foodReach = {}

    for pos in food.asList():
      foodReach[pos] = self.getMazeDistance(teamPos, pos) - self.getMazeDistance(enemyPos, pos)

    return foodReach

  def onOurSide(self, gameState, position):
    if self.red:
      return gameState.isRed(position)
    else:
      return not gameState.isRed(position)

  def impossiblePositionBelief(self, gameState, enemyIndex, position):
    isPacman = gameState.getAgentState(enemyIndex).isPacman
    onSide = self.onOurSide(gameState, position)
    return (isPacman and not onSide) or (not isPacman and onSide)

  def initializeUniformly(self, enemyIndex, gameState):
    global beliefs
    global validPositions
    global SIGHT_RANGE

    team = [gameState.getAgentPosition(index) for index in self.getTeam(gameState)]
    newBelief = util.Counter()
    enemyPacman = gameState.getAgentState(enemyIndex).isPacman

    for position in validPositions:
      invalid = self.impossiblePositionBelief(gameState, enemyIndex, position) or self.inRangeOfAny(team, position, SIGHT_RANGE)

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
    return util.manhattanDistance(pos1, pos2) <= dist

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
    return Actions.getLegalNeighbors(pos, gameState.getWalls())

  def establishLocation(self, enemyIndex, pos):
    global beliefs

    beliefs[enemyIndex] = util.Counter()
    beliefs[enemyIndex][pos] = 1.0
    beliefs[enemyIndex].normalize()

  def updatePotentiallyEatenEnemies(self, gameState):
    for enemyIndex in self.getOpponents(gameState):
      if self.wasEaten(enemyIndex):
        self.establishLocation(enemyIndex, gameState.getInitialAgentPosition(enemyIndex))

  def wasEaten(self, enemyIndex):
    if enemyIndex not in self.lastPositions:
      return False
    lastPos = self.lastPositions[enemyIndex]
    currentPos = self.lastPositions[enemyIndex]

    return lastPos is not None and currentPos is None

  def observe(self, gameState):
    global beliefs
    global validPositions
    global SIGHT_RANGE

    # Noisy Agent Distances
    agentDists = gameState.getAgentDistances()
    myPos = gameState.getAgentPosition(self.index)
    team = [gameState.getAgentPosition(index) for index in self.getTeam(gameState)]

    for enemyIndex in self.getOpponents(gameState):
      assert sum(beliefs[enemyIndex].values()) >= .99999
      
      newBelief = util.Counter()
      enemyPosition = gameState.getAgentPosition(enemyIndex)

      if enemyPosition is not None:
        self.establishLocation(enemyIndex, enemyPosition)
        continue
      else:
        noisyDistance = agentDists[enemyIndex]

        for position in beliefs[enemyIndex]:

          if self.impossiblePositionBelief(gameState, enemyIndex, position) or self.inRangeOfAny(team, position, SIGHT_RANGE):
            continue

          trueDistance = util.manhattanDistance(myPos, position)
          distanceProb = gameState.getDistanceProb(trueDistance, noisyDistance)

          positionBelief = beliefs[enemyIndex][position] * distanceProb
          newBelief[position] = positionBelief

      if sum(newBelief.values()) == 0:
        self.initializeUniformly(enemyIndex, gameState)
        # self.updatePotentiallyEatenEnemies(gameState)

        # if sum(newBelief.values()) == 0:
        #   assert False
      else:
        newBelief.normalize()
        beliefs[enemyIndex] = newBelief

      assert sum(beliefs[enemyIndex].values()) >= .99999

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
      self.agentFoodEaten[mostLikelyIndices[index]] += 1

  def elapseTime(self, gameState):
    global beliefs
    global validPositions
    global SIGHT_RANGE

    newBeliefs = {}
    team = [gameState.getAgentPosition(index) for index in self.getTeam(gameState)]

    for enemyIndex in self.getOpponents(gameState):
      
      if (enemyIndex + 1) % gameState.getNumAgents() != self.index:
        continue

      newBeliefs[enemyIndex] = util.Counter()

      for oldPos in beliefs[enemyIndex]:
        if beliefs[enemyIndex][oldPos] == 0.0:
          continue

        newPositions = self.getNextPositions(gameState, oldPos)

        for newPos in newPositions:
          if newPos in validPositions or self.impossiblePositionBelief(gameState, enemyIndex, newPos) or self.inRangeOfAny(team, newPos, SIGHT_RANGE):
            if newPos not in newBeliefs[enemyIndex]:
              newBeliefs[enemyIndex][newPos] = 0.0

            newBeliefs[enemyIndex][newPos] += (beliefs[enemyIndex][oldPos] * 1.0) / len(newPositions)
      
      newBeliefs[enemyIndex].normalize()
      beliefs[enemyIndex] = newBeliefs[enemyIndex]

  def anyInvalidPositions(self):
    global beliefs
    global validPositions

    for enemyIndex in beliefs:
      for pos in beliefs[enemyIndex]:
        if pos not in validPositions:
          return True

    return False

  def getDistributions(self, gameState):
    global beliefs
    
    distributions = []
    opponents = self.getOpponents(gameState)
    mostLikelyPositions = [self.getMostLikelyPosition(enemyIndex, gameState) for enemyIndex in self.getOpponents(gameState)]

    for index in xrange(gameState.getNumAgents()):
      if index in opponents:
        distributions.append(beliefs[index].copy())
      else:
        distributions.append(None)

    if len(self.distributions) > 0:
      distributions[opponents[0]] = self.distributions.pop()

    blocks = self.optimalBlocks.copy() if self.optimalBlocks is not None else util.Counter()
    for pos in blocks:
      if blocks[pos] < 3:
        blocks[pos] = 0.0
    distributions.append(blocks)

    if self.trappedFood is not None:
      distributions.append(self.trappedFood.copy())

    # if self.bestDefensePositions is not None:
    #   defense = self.bestDefensePositions.copy()
    #   for value in defense:
    #     if defense[value] != 0:
    #       defense[value] = 1.0 / defense[value] ** 2
    #   defense.normalize()
    #   distributions.append(defense)
    #for value in self.foodDistributions:
      #foodDist = self.foodDistributions[value]
      #maxima = max(foodDist.values())
      #bestOptions = [pos for pos in foodDist if foodDist[pos] == maxima]
      #bestOption = bestOptions[0]

      #newDist = util.Counter()
      #newDist[bestOption] = 1.0
      #distributions.append(newDist)


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
          self.foodDistributions[team][pos] = 1.0 / (self.getSumSquareDistances(pos, self.getFoodYouAreDefending(gameState)) + 0.000000001)
        else:
          self.foodDistributions[team][pos] = 1.0 / (self.getSumSquareDistances(pos, self.getFood(gameState)) + 0.000000001)
      self.foodDistributions[team].normalize()

      maxima = max(self.foodDistributions[team].values())
      positions = [position for position in self.foodDistributions[team].keys() if self.foodDistributions[team][position] == maxima]
      self.foodDistributions[team] = util.Counter()
      self.foodDistributions[team][positions[0]] = 1.0

  def getOptimalFoodPosition(self, gameState, isRed):
    if isRed == self.red:
      dist = self.foodDistributions[self.red]
    else:
      dist = self.foodDistributions[not self.red]

    values = dist.values()
    if len(values) == 0:
      return None

    maxima = max(values)
    positions = [pos for pos in dist if dist[pos] == maxima]
    return positions[0]

  def getMSTWeight(self, points):
    weight = 0.0
    node = points[0]

    while len(points) > 1:
      points.remove(node)

      pointDists = [self.getMazeDistance(node, point) for point in points]
      bestDist = min(pointDists)
      bestPoints = [point for point in points if self.getMazeDistance(node, point) == bestDist]

      weight += bestDist
      node = bestPoints[0]

    return weight

  def dfs2(self, gameState, position, depth, seen=[], goodPositions=[]):
    if depth == 0 or position in seen:
      return 0

    seen.append(position)
    positions = position in goodPositions

    for pos in self.getNextPositions(gameState, position):
      positions += self.dfs2(gameState, pos, depth - 1, seen, goodPositions)

    return positions

  def dfsTilFork(self, gameState, position, seen=[]):
    if position in seen:
      return []
    nextPositions = self.getNextPositions(gameState, position)

    if len(nextPositions) > 3:
      return [position]

    forks = []
    seen.append(position)
    for pos in nextPositions:
       forks += self.dfsTilFork(gameState, pos, seen)
    
    return forks

  def discoverOptimalBlock(self, gameState):
    defFood = self.getFoodYouAreDefending(gameState).asList()
    forkCount = util.Counter()
    
    for food in defFood:
      forks = self.dfsTilFork(gameState, food, [])
      if len(forks) == 1:
        forkCount[forks[0]] += 1

    if any(forkCount):
      return forkCount

    return None

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    self.elapseTime(gameState)
    self.observe(gameState)
    self.setFoodDistanceDistribution(gameState)

    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
    self.updatePotentiallyEatenEnemies(gameState)
    distributions = self.getDistributions(gameState)

    self.displayDistributionsOverPositions(distributions)
    
    self.lastFood = self.getFoodYouAreDefending(gameState)
    self.lastPositions = {}
    for index in xrange(gameState.getNumAgents()):
      self.lastPositions[index] = gameState.getAgentPosition(index)

    if gameState.getAgentPosition(self.index) not in self.positionCount:
        self.positionCount[gameState.getAgentPosition(self.index)] = 0
    if not self.onOurSide(gameState, gameState.getAgentPosition(self.index)):
        self.positionCount[gameState.getAgentPosition(self.index)] += 1

    for index in self.getOpponents(gameState):
      if gameState.getAgentState(index).isPacman:
        self.pacmanTime[index] += 1

    action = random.choice(bestActions)
    self.lastMoves.append(gameState.getAgentPosition(self.index))

    return action

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

    # print self, action
    # for feature in features:
      # print feature, "\t", features[feature]
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

########################### UTILITIES ################################

  def getTeammateIndex(self, gameState):
    """ Returns your teammates index assuming you have 1 teammate"""
    indices = self.getTeam(gameState)
    indices.remove(self.index)
    return indices[0]

  def dfs(self, gameState, position, seen, func, depth=-1):
    """Returns a list of positions within depth and terminating on func's true return."""
    if position in seen or depth == 0:
      return []
    
    if func(position):
      return [position]

    if depth > 0:
      depth = depth - 1

    seen.append(position)
    discoveredPositions = []

    nextPositions = self.getNextPositions(gameState, position)

    for nextPosition in nextPositions:
      discoveredPositions += self.dfs(gameState, nextPosition, seen, func, depth)

    return discoveredPositions

######################## SPECIAL UTILITIES ############################

  def getTeamValidPositions(self, gameState):
    teamValidPositions = []

    for position in validPositions:
      if self.onOurSide(gameState, position):
        teamValidPositions.append(position)

    return teamValidPositions

  def getEnemyValidPositions(self, gameState):
    enemyValidPositions = []

    for position in validPositions:
      if not self.onOurSide(gameState, position):
        enemyValidPositions.append(position)

    return enemyValidPositions

  def getTeamWall(self, gameState):
    teamValidPositions = self.getTeamValidPositions(gameState)
    teamValidPositions = sorted(teamValidPositions, key=lambda pos: pos[0])

    if self.red:
      teamValidPositions.reverse()

    wallXCoordinate = teamValidPositions[0][0]
    teamWall = [coord for coord in teamValidPositions if coord[0] == wallXCoordinate]

    return teamWall

  def getEnemyWall(self, gameState):
    enemyValidPositions = self.getEnemyValidPositions(gameState)
    enemyValidPositions = sorted(enemyValidPositions, key=lambda pos: pos[0])

    if not self.red:
      enemyValidPositions.reverse()

    wallXCoordinate = enemyValidPositions[0][0]
    enemyWall = [coord for coord in enemyValidPositions if coord[0] == wallXCoordinate]
    
    return enemyWall

  def getFoodExitDict(self, gameState, foodList):
    foodExitDict = {}

    for food in foodList:
      foodExitDict[food] = self.dfs(gameState, food, list(), func=lambda x: len(self.getNextPositions(gameState, x)) >= 4, depth=-1)

      # We don't these pieces of food to be considered trapped
      if food in foodExitDict[food]:
        foodExitDict[food].remove(food)

    return foodExitDict

  def getFreeFood(self, foodExitDict):
    freeFood = []

    for food in foodExitDict:
      if len(foodExitDict[food]) != 1:
        freeFood.append(food)

    return freeFood

  def getTrappedFood(self, foodExitDict):
    trappedFood = []

    for food in foodExitDict:
      if len(foodExitDict[food]) == 1:
        trappedFood.append(food)

    return trappedFood

  def distanceToPath(self, gameState, pos1, pos2, badPositions=[]):
    seen = []

    pQueue = util.PriorityQueueWithFunction(lambda pos_cost: pos_cost[1] + self.getMazeDistance(pos_cost[0], pos2))
    pQueue.push((pos1, 0))

    while not pQueue.isEmpty():
      position, cost = pQueue.pop()

      if position == pos2:
        return cost

      if position in seen or position in badPositions:
        continue

      seen.append(position)
      for nextPosition in self.getNextPositions(gameState, position):
        pQueue.push((nextPosition, cost + 1))

    # Couldn't find such a path
    return 1e10

#######################################################################


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):

    ######################## VARIABLES #########################

    successor = self.getSuccessor(gameState, action)

    oldPos = gameState.getAgentPosition(self.index)
    newPos = successor.getAgentPosition(self.index)

    isPacman = successor.getAgentState(self.index).isPacman

    enemyValidPositions = self.getEnemyValidPositions(gameState)
    teamValidPositions = self.getTeamValidPositions(gameState)

    enemyWall = self.getEnemyWall(gameState)
    teamWall = self.getTeamWall(gameState)

    enemyFood = self.getFood(gameState).asList()

    enemyFoodDict = self.getFoodExitDict(gameState, enemyFood)

    trappedEnemyFood = self.getTrappedFood(enemyFoodDict)

    self.trappedFood = util.Counter()
    for food in trappedEnemyFood:
      self.trappedFood[food] = 1.0
    self.trappedFood.normalize()

    freeEnemyfood = self.getFreeFood(enemyFoodDict)

    enemyIndices = self.getOpponents(gameState)
    enemyStates = [gameState.getAgentState(index) for index in enemyIndices]
    enemyPacmen = [index for index in enemyIndices if gameState.getAgentState(index).isPacman]
    enemyGhosts = [index for index in enemyIndices if not gameState.getAgentState(index).isPacman]

    teamIndices = self.getTeam(gameState)
    teamStates = [gameState.getAgentState(index) for index in teamIndices]
    teamPacmen = [index for index in teamIndices if gameState.getAgentState(index).isPacman]
    teamGhosts = [index for index in teamIndices if not gameState.getAgentState(index).isPacman]

    """ Inference Variables (mixed) """

    enemyPositions = [self.getMostLikelyPosition(index, gameState) for index in enemyIndices]

    #############################################################

    """ FEATURES """

    features = util.Counter()
    myState = gameState.getAgentState(self.index)

    mostLikelyInvaderPositions = [self.getMostLikelyPosition(index, gameState) for index in self.getOpponents(gameState) if gameState.getAgentState(index).isPacman]
    mostLikelyGhostPositions = [self.getMostLikelyPosition(index, gameState) for index in self.getOpponents(gameState) if not gameState.getAgentState(index).isPacman]
    enemyStates = [(index, gameState.getAgentState(index)) for index in self.getOpponents(gameState)]
    pacmen = [(index, state) for (index, state) in enemyStates if state.isPacman]

    for invader in mostLikelyInvaderPositions:
      dist = self.getMazeDistance(invader, newPos)
      if dist <= SIGHT_RANGE and not myState.isPacman and not len(pacmen) == len(enemyStates):
        features["invaderDistance"] = dist
        return features

    if len(enemyPacmen) == len(enemyIndices):
      foodDists = [self.getMazeDistance(newPos, food) for food in enemyFood]
      features["min-food-distance"] = min(foodDists)
      return features

    # Offensive Juking - prevents immediate death
    features["1-dist-ghost-count"] = 0

    for nextPosition in self.getNextPositions(gameState, newPos):
      for enemyIndex in enemyIndices:
        if gameState.getAgentState(enemyIndex).isPacman:
          continue
        enemyPosition = self.getMostLikelyPosition(enemyIndex, gameState)
        if nextPosition == enemyPosition:
          features["1-dist-ghost-count"] += 1

    if not isPacman:
      for enemyIndex in enemyPacmen:
        enemyPosition = self.getMostLikelyPosition(enemyIndex, gameState)

        if self.getMazeDistance(newPos, enemyPosition) <= SIGHT_RANGE:
          oldDistanceToInvader = self.getMazeDistance(oldPos, enemyPosition)
          newDistanceToInvader = self.getMazeDistance(newPos, enemyPosition)

          # The more positive the better
          features["dist-to-invader"] += (oldDistanceToInvader - newDistanceToInvader)

    badPositions = enemyPositions[:]

    # Attack the free enemy food if any left.
    if len(freeEnemyfood) > 0:

      # Sort food from what was closest previously.
      closestFreeEnemyFoodList = sorted(freeEnemyfood, key=lambda food: self.distanceToPath(gameState, oldPos, food, badPositions))

      self.distributions = []
      freeFoodDist = util.Counter()
      for index, food in enumerate(closestFreeEnemyFoodList):
        freeFoodDist[food] = 1.0 / (index + 1.0)
      freeFoodDist.normalize()
      self.distributions.append(freeFoodDist)

      if len(closestFreeEnemyFoodList) > 0:
        closestFreeEnemyFood = closestFreeEnemyFoodList[0]

        oldDistance = self.distanceToPath(gameState, closestFreeEnemyFood, oldPos, enemyPositions)
        newDistance = self.distanceToPath(gameState, closestFreeEnemyFood, newPos, enemyPositions)

        # We want values to be binary, so we take the difference in distances.
        # The more positive, the better.
        features["free-food-distance"] = oldDistance - newDistance

    # Eat closest trapped food relative to your old position
    # if your belief to survive is strong.
    if len(trappedEnemyFood) > 0:
      closestTrappedFoodList = sorted(trappedEnemyFood, key=lambda food: self.distanceToPath(gameState, oldPos, food, badPositions))
      closestTrappedFood = closestTrappedFoodList[0]

      # The exit should be placed in the corresponding food counter,
      # and there should only be one value
      exit = enemyFoodDict[closestTrappedFood][0]
      
      oldDistToTrappedFood = self.distanceToPath(gameState, oldPos, closestTrappedFood, badPositions)
      newDistToTrappedFood = self.distanceToPath(gameState, newPos, closestTrappedFood, badPositions)
      distFromFoodToExit = self.getMazeDistance(closestTrappedFood, exit)

      # We have to go in and out if we want to succeed
      oldTotalMoveDistance = oldDistToTrappedFood + distFromFoodToExit
      newTotalMoveDistance = newDistToTrappedFood + distFromFoodToExit

      enemyDistancesFromExit = [self.getMazeDistance(exit, enemyPos) for enemyPos in enemyPositions]
      closestEnemyDistance = min(enemyDistancesFromExit)

      # The more positive, the better.
      if closestEnemyDistance > newTotalMoveDistance:
        features["trapped-food-distance"] = oldTotalMoveDistance - newTotalMoveDistance
      elif len(freeEnemyfood) == 0: # If we can't do anything else, find the closest slot to take
          features["trapped-food-distance"] = oldDistToTrappedFood - newDistToTrappedFood

    return features

  def getWeights(self, gameState, action):
    return {
      'successorScore': 1000,
      'reverse': -10,
      'offense': 100,
      'tooClose': -1000,
      'stop': -10,
      'foodDistance': 10,
      'closestFood': -10,
      'invaderDistance': -10,
      'samePosition': -2,
      'ghost-distance': 5,
      '1-dist-ghost-count': -1000,
      'trapped-food-distance': 150,
      'free-food-distance': 100,
      'distance-from-wall': 20,
      'distance-from-team-wall': 10,
      'distance-from-enemy-wall': -10,
      'min-food-distance': -10,
      'dist-to-invader': 15,
    }

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
    oldPos = gameState.getAgentPosition(self.index)
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    """ NEW FEATURES """
    teamWall = self.getTeamWall(gameState)

    believedPositions = []
    for enemyIndex in beliefs:
      believedPositions += beliefs[enemyIndex].keys()

    defensePositions = self.foodDistributions[self.red]
    maxima = max(defensePositions.values())
    bestOptions = [pos for pos in defensePositions if defensePositions[pos] == maxima]
    bestDefenseOption = bestOptions[0]

    mostLikelyEnemyPositions = [self.getMostLikelyPosition(index, gameState) for index in self.getOpponents(gameState)]
    enemyStates = [(index, gameState.getAgentState(index)) for index in self.getOpponents(gameState)]
    pacmen = [(index, state) for (index, state) in enemyStates if state.isPacman]

    foodDistances = {}

    for enemy in mostLikelyEnemyPositions:
      foodDistances[enemy] = [(food, self.getMazeDistance(enemy, food)) for food in self.getFoodYouAreDefending(successor).asList()]

    foodEaten = 0
    for enemyIndex in self.agentFoodEaten:
        foodEaten += self.agentFoodEaten[enemyIndex]

    minFoodEaten = 0
    maxFoodEaten = 0
    heaviestEater = None
    if foodEaten > 0:
      maxFoodEaten = max(self.agentFoodEaten.values())
      minFoodEaten = min(self.agentFoodEaten.values())
      for enemyIndex in self.agentFoodEaten:
          if self.agentFoodEaten[enemyIndex] == maxFoodEaten:
              heaviestEater = enemyIndex
              break

    if heaviestEater is None and sum(self.pacmanTime.values()) != 0.0:
      maxTime = max(self.pacmanTime.values())
      indices = [index for index in self.getOpponents(gameState) if self.pacmanTime[index] == maxTime]

      pacmanIndex = indices[0]
      heaviestEater = pacmanIndex

    if len(pacmen) == 0 and heaviestEater is None:
      enemyPositionUsed = mostLikelyEnemyPositions[0]
      closestFood = sorted(foodDistances[enemyPositionUsed], key=lambda x: x[1])[0][0]
      oldDistance = self.getMazeDistance(oldPos, closestFood)
      newDistance = self.getMazeDistance(myPos, closestFood)
      features["closestFood"] = oldDistance - newDistance
      features["closestEnemy"] = sum([self.getMazeDistance(pos, myPos) for pos in mostLikelyEnemyPositions])
    elif len(pacmen) == len(mostLikelyEnemyPositions):
      forkCount = self.discoverOptimalBlock(gameState)
      maxCount = 0
      minCount = 0
      if forkCount is not None:
         self.optimalBlocks = forkCount.copy()
         maxCount = max(forkCount.values())
      if maxCount >= 3:
          if self.lastBestBlock is None or forkCount[self.lastBestBlock] != maxCount:
            defenseOptions = [pos for pos in forkCount if forkCount[pos] >= 3]
            defenseOptions = sorted(defenseOptions, key=lambda x: self.getMazeDistance(myPos, x))
            bestDefenseSpot = defenseOptions[0]
            self.lastBestBlock = bestDefenseSpot
          else:
            bestDefenseSpot = self.lastBestBlock
          features = util.Counter()
          features["onDefense"] = 1.0
          features["finalDefense"] = self.getMazeDistance(bestDefenseSpot, myPos)
      else:
          features["optimalDefenseDistance"] = self.getMazeDistance(myPos, bestDefenseOption)
          if len(self.getFoodYouAreDefending(gameState).asList()) < 10:
              features["optimalDefenseDistance"] = features["optimalDefenseDistance"] ** .75
          pacmenPositions = [self.getMostLikelyPosition(index, gameState) for index, state in pacmen]
          pacmenDistances = [self.getMazeDistance(pos, myPos) for pos in pacmenPositions]
          if len(pacmenDistances) > 0:
            features["closestEnemy"] = min(pacmenDistances)
    else:
      if heaviestEater is not None:
        trackedEnemy = self.getMostLikelyPosition(heaviestEater, gameState)
        defSide = [pos for pos in validPositions if self.onOurSide(gameState, pos)]
        bestPositions = [(pos, self.getMazeDistance(pos, trackedEnemy)) for pos in defSide]
        bestPositions = sorted(bestPositions, key=lambda x: x[1])
        bestDefensiveTrackedPosition = bestPositions[0][0]
        self.bestDefensePositions = util.Counter()
        for pos, dist in bestPositions:
          self.bestDefensePositions[pos] = dist

      pacmenPositions = [self.getMostLikelyPosition(index, gameState) for index, state in pacmen]
      pacmenDistances = [self.getMazeDistance(pos, myPos) for pos in pacmenPositions]
      features = util.Counter()
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0
      if len(pacmenDistances) > 0:
        features["closestEnemy"] = min(pacmenDistances)
      if heaviestEater is not None:
        features["closestEater"] = self.getMazeDistance(myPos, bestDefensiveTrackedPosition)

    return features

  def getWeights(self, gameState, action):
    return {
      'finalDefense': -100000,
      'onDefense': 100000000,
      "closestEnemy": -30,
      'optimalDefenseDistance': -5,
      'closestFood': 10,
      'closestEater': -15,
      'closestFoodWhileTracking': -1,
    }

