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
class HumblestAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    return ReflexCaptureAgent(index)

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

  def initializeLasts(self):
    self.last = {}
    self.last["enemyFood"] = None
    self.last["teamFood"] = None

  def updateLasts(self, gameState):
    self.last["enemyFood"] = self.getFood(gameState)
    self.last["teamFood"] = self.getFoodYouAreDefending(gameState)

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    global beliefs
    global validPositions

    self.initializeLasts()
    self.enemyFoodCount = {}

    for enemyIndex in self.getOpponents(gameState):
      self.enemyFoodCount[enemyIndex] = 0

    self.variousDistributions = []

    if len(validPositions) == 0:
      # All positions are those that are not walls
      validPositions = gameState.getWalls().asList(False)

      # We know that each enemy must be at its initial position at registration
      for enemyIndex in self.getOpponents(gameState):
        self.establishLocation(enemyIndex, gameState.getInitialAgentPosition(enemyIndex))

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

  def getMostLikelyEnemies(self, gameState, pos):
    dists = [(index, self.getMazeDistance(pos, self.getMostLikelyPosition(index, gameState))) for index in self.getOpponents(gameState)]
    sortedIndices = [index for index, dist in sorted(dists, key=lambda x: x[1])]

    candidateValues = []

    # Assumes beliefs contain positions that are only nonzero
    for enemy in self.getOpponents(gameState):
      beliefPositions = beliefs[enemy].keys()

      # Score on belief of that piece being there
      if pos in beliefPositions:
        candidateValues.append((enemy, beliefs[enemy][pos]))

    
    if len(candidateValues) > 0:
      candidateValues = sorted(candidateValues, key=lambda x: x[1])
      candidateValues.reverse()
      sortedCandidateIndices = [candidateIndex for candidateIndex, prob in candidateValues]
      return sortedCandidateIndices + [index for index in sortedIndices if index not in sortedCandidateIndices]

    return sortedIndices

  def getMostLikelyPosition(self, index, gameState):
    if index in self.getTeam(gameState):
      return gameState.getAgentPosition(index)
    else:
      maxProb = max(beliefs[index].values())
      bestPositions = [position for position in beliefs[index].keys() if beliefs[index][position] == maxProb]
      return random.choice(bestPositions)

  def getMissingFoodPositions(self, gameState):
    oldFoodList = self.last["teamFood"].asList()
    currentFoodList = self.getFoodYouAreDefending(gameState).asList()

    eatenFood = []

    for food in oldFoodList:
      if food not in currentFoodList:
        eatenFood.append(food)

    return eatenFood

  def getNextPositions(self, gameState, pos):
    return Actions.getLegalNeighbors(pos, gameState.getWalls())

  def establishLocation(self, enemyIndex, pos):
    beliefs[enemyIndex] = util.Counter()
    beliefs[enemyIndex][pos] = 1.0
    beliefs[enemyIndex].normalize()

  def observe(self, gameState):
    agentDists = gameState.getAgentDistances()
    myPos = gameState.getAgentPosition(self.index)
    team = [gameState.getAgentPosition(index) for index in self.getTeam(gameState)]

    for enemyIndex in self.getOpponents(gameState):      
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
      else:
        newBelief.normalize()
        beliefs[enemyIndex] = newBelief

    if self.last["teamFood"] == None:
      return

    eatenFood = self.getMissingFoodPositions(gameState)
    chosenIndices = []

    for food in eatenFood:
      mostLikelyIndices = self.getMostLikelyEnemies(gameState, food)
      index = 0

      while mostLikelyIndices[index] in chosenIndices:
        index += 1

      mostLikelyEnemyIndex = mostLikelyIndices[index]
      chosenIndices.append(mostLikelyEnemyIndex)
      self.establishLocation(mostLikelyEnemyIndex, food)
      self.enemyFoodCount[mostLikelyEnemyIndex] += 1

  def elapseTime(self, gameState):
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

  def getDistributions(self, gameState):    
    distributions = [None for _ in xrange(gameState.getNumAgents())]

    for enemyIndex in self.getOpponents(gameState):
      distributions[enemyIndex] = beliefs[enemyIndex].copy()

    for dist in self.variousDistributions:
      distributions.append(dist.copy())

    return distributions

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    self.elapseTime(gameState)
    self.observe(gameState)

    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
    distributions = self.getDistributions(gameState)
    self.displayDistributionsOverPositions(distributions)
    self.updateLasts(gameState)

    action = random.choice(bestActions)
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
    if self.index < self.getTeammateIndex(gameState):
      features = self.getOffenseFeatures(gameState, action)
      weights = self.getOffenseWeights(gameState, action)
    else:
      features = self.getDefenseFeatures(gameState, action)
      weights = self.getDefenseWeights(gameState, action)
    return features * weights

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

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getDefenseFeatures(self, gameState, action):

    ######################## VARIABLES #########################

    successor = self.getSuccessor(gameState, action)

    oldPos = gameState.getAgentPosition(self.index)
    newPos = successor.getAgentPosition(self.index)

    wasPacman = gameState.getAgentState(self.index).isPacman
    isPacman = successor.getAgentState(self.index).isPacman

    enemyValidPositions = self.getEnemyValidPositions(gameState)
    teamValidPositions = self.getTeamValidPositions(gameState)

    enemyWall = self.getEnemyWall(gameState)
    teamWall = self.getTeamWall(gameState)

    enemyFood = self.getFood(gameState).asList()
    teamFood = self.getFoodYouAreDefending(gameState).asList()
    allFood = enemyFood + teamFood

    enemyFoodLeft = len(enemyFood)
    teamFoodLeft =len(teamFood)

    enemyFoodDict = self.getFoodExitDict(gameState, enemyFood)
    teamFoodDict = self.getFoodExitDict(gameState, teamFood)
    allFoodDict = self.getFoodExitDict(gameState, allFood)

    trappedEnemyFood = self.getTrappedFood(enemyFoodDict) 
    trappedTeamFood = self.getTrappedFood(teamFoodDict)
    trappedAllFood = self.getTrappedFood(allFoodDict)

    freeEnemyfood = self.getFreeFood(enemyFoodDict)
    freeTeamFood = self.getFreeFood(teamFoodDict)
    freeAllFood = self.getFreeFood(allFoodDict)

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
    teamPositions = [gameState.getAgentPosition(index) for index in teamIndices]

    #############################################################

    """ FEATURES """

    features = util.Counter()
    features["on-defense"] = 1 if not isPacman else 0

    # If they're on full offense, retreat to defense
    if len(enemyPacmen) == len(enemyIndices):

      # Try to find the best location to block at least 3 pellets.
      exitMap = util.Counter()

      for food in trappedTeamFood:
        exits = teamFoodDict[food]
        for exit in exits:
          exitMap[exit] += 1

      blockingOptions = [exit for exit in exitMap if exitMap[exit] >= 3]
      blockingOptions = sorted(blockingOptions, key=lambda exit: self.getMazeDistance(oldPos, exit))

      if len(blockingOptions) > 0:
        bestBlockingPosition = blockingOptions[0]

        oldDistanceToBlock = self.getMazeDistance(oldPos, bestBlockingPosition)
        newDistanceToBlock = self.getMazeDistance(newPos, bestBlockingPosition)

        # The more positive, the better
        features["distance-to-block"] = oldDistanceToBlock - newDistanceToBlock
        return features
      else:

        pos_distance = []
        # Try to minimize your distance to your pieces
        # Use sqrt distance to stay near groups of pellets
        for pos in teamValidPositions:
          dist = 0.0
          for food in teamFood:
            dist += self.getMazeDistance(pos, food) ** .75
          pos_distance.append((food, dist))

        pos_distance = sorted(pos_distance, key=lambda pos_dist: pos_dist[1])
        bestDefensivePosition = pos_distance[0][0]

        # print bestDefensivePosition
        oldDistanceToDefensivePos = self.getMazeDistance(oldPos, bestDefensivePosition)
        newDistanceToDefensivePos = self.getMazeDistance(newPos, bestDefensivePosition)

        # The more positive, the better
        features["distance-to-defend"] = newDistanceToDefensivePos

        features["minimize-invader-distance"] = 0.0

        for enemyIndex in enemyPacmen:
          mostLikelyInvaderPosition = self.getMostLikelyPosition(enemyIndex, gameState)

          oldDistanceToDangerousInvader = self.getMazeDistance(oldPos, mostLikelyInvaderPosition)
          newDistanceToDangerousInvader = self.getMazeDistance(newPos, mostLikelyInvaderPosition)

          # The more positive, the better
          features["minimize-invader-distance"] += (oldDistanceToDangerousInvader - newDistanceToDangerousInvader)
        return features

    # Minimize distance to agent who has eaten the most - if tied, minimize both
    enemy_food = [(enemyIndex, self.enemyFoodCount[enemyIndex]) for enemyIndex in  enemyIndices]
    mostFoodEaten = max([foodCount for index, foodCount in enemy_food])
    mostDangerousEnemies = [enemyIndex for enemyIndex, foodCount in enemy_food if foodCount == mostFoodEaten]

    if len(enemyPacmen) > 0:
      features["minimize-invader-distance"] = 0.0

      for enemyIndex in enemyPacmen:
        mostLikelyInvaderPosition = self.getMostLikelyPosition(enemyIndex, gameState)

        oldDistanceToDangerousInvader = self.getMazeDistance(oldPos, mostLikelyInvaderPosition)
        newDistanceToDangerousInvader = self.getMazeDistance(newPos, mostLikelyInvaderPosition)

        # The more positive, the better
        features["minimize-invader-distance"] += (oldDistanceToDangerousInvader - newDistanceToDangerousInvader)

        # Stay closer to our side to protect pieces
        # The more positive, the better
        enemyWallXCoordinate = enemyWall[0][0]
        features["distance-from-enemy-wall"] = newPos[0] - enemyWallXCoordinate
    else:
      features["minimize-enemy-distance"] = 0.0
      for enemyIndex in mostDangerousEnemies:
        mostLikelyEnemyPosition = self.getMostLikelyPosition(enemyIndex, gameState)

        oldDistanceToDangerousEnemy = self.getMazeDistance(oldPos, mostLikelyEnemyPosition)
        newDistanceToDangerousEnemy = self.getMazeDistance(newPos, mostLikelyEnemyPosition)

        # The more positive, the better
        features["minimize-enemy-distance"] += (oldDistanceToDangerousEnemy - newDistanceToDangerousEnemy)

    return features

  def getOffenseFeatures(self, gameState, action):

    ######################## VARIABLES #########################

    successor = self.getSuccessor(gameState, action)

    oldPos = gameState.getAgentPosition(self.index)
    newPos = successor.getAgentPosition(self.index)

    wasPacman = gameState.getAgentState(self.index).isPacman
    isPacman = successor.getAgentState(self.index).isPacman

    enemyValidPositions = self.getEnemyValidPositions(gameState)
    teamValidPositions = self.getTeamValidPositions(gameState)

    enemyWall = self.getEnemyWall(gameState)
    teamWall = self.getTeamWall(gameState)

    enemyFood = self.getFood(gameState).asList()
    teamFood = self.getFoodYouAreDefending(gameState).asList()
    allFood = enemyFood + teamFood

    enemyFoodLeft = len(enemyFood)
    teamFoodLeft =len(teamFood)

    enemyFoodDict = self.getFoodExitDict(gameState, enemyFood)
    teamFoodDict = self.getFoodExitDict(gameState, teamFood)
    allFoodDict = self.getFoodExitDict(gameState, allFood)

    trappedEnemyFood = self.getTrappedFood(enemyFoodDict) 
    trappedTeamFood = self.getTrappedFood(teamFoodDict)
    trappedAllFood = self.getTrappedFood(allFoodDict)

    freeEnemyfood = self.getFreeFood(enemyFoodDict)
    freeTeamFood = self.getFreeFood(teamFoodDict)
    freeAllFood = self.getFreeFood(allFoodDict)

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
    teamPositions = [gameState.getAgentPosition(index) for index in teamIndices]

    #############################################################

    """ FEATURES """

    features = util.Counter()

    if len(enemyPacmen) == len(enemyIndices):
      foodDists = [self.getMazeDistance(newPos, food) for food in enemyFood]
      features["min-food-distance"] = min(foodDists)
      return features

    # Prevent death
    features["initial-position"] = newPos == gameState.getInitialAgentPosition(self.index)

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

    # Attack the free enemy food if any left.
    if len(freeEnemyfood) > 0:

      # Sort food from what was closest previously.
      closestFreeEnemyFoodList = sorted(freeEnemyfood, key=lambda food: self.distanceToPath(gameState, oldPos, food, enemyPositions))
      closestFreeEnemyFood = closestFreeEnemyFoodList[0]

      oldDistance = self.distanceToPath(gameState, closestFreeEnemyFood, oldPos, enemyPositions)
      newDistance = self.distanceToPath(gameState, closestFreeEnemyFood, newPos, enemyPositions)

      # We want values to be binary, so we take the difference in distances.
      # The more positive, the better.
      features["free-food-distance"] = oldDistance - newDistance

    # Eat closest trapped food relative to your old position
    # if your belief to survive is strong.
    if len(trappedEnemyFood) > 0:
      closestTrappedFoodList = sorted(trappedEnemyFood, key=lambda food: self.distanceToPath(gameState, oldPos, food, enemyPositions))
      closestTrappedFood = closestTrappedFoodList[0]

      # The exit should be placed in the corresponding food counter,
      # and there should only be one value
      exit = enemyFoodDict[closestTrappedFood][0]
      
      oldDistToTrappedFood = self.distanceToPath(gameState, oldPos, closestTrappedFood, enemyPositions)
      newDistToTrappedFood = self.distanceToPath(gameState, newPos, closestTrappedFood, enemyPositions)
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

    if isPacman:
      # The more positive, the better
      teamWallXCoordinate = teamWall[0][0]
      features["distance-from-team-wall"] = newPos[0] - teamWallXCoordinate
    else:
      # The more negative, the better
      enemyWallXCoordinate = enemyWall[0][0]
      features["distance-from-enemy-wall"] = newPos[0] - enemyWallXCoordinate

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}

  def getDefenseWeights(self, gameState, action):
    return {
      'on-defense': 1e5,
      'distance-to-block': 1e10,
      'distance-to-defend': -500,
      'minimize-invader-distance': 75,
      'minimize-enemy-distance': 50,
      'distance-from-enemy-wall': -100,
    }

  def getOffenseWeights(self, gameState, action):
    return {
      'initial-position': -1e10,
      '1-dist-ghost-count': -1000,
      'trapped-food-distance': 150,
      'free-food-distance': 100,
      'distance-from-wall': 20,
      'distance-from-team-wall': 10,
      'distance-from-enemy-wall': -10,
      'min-food-distance': -2,
      'dist-to-invader': 15,
    }
