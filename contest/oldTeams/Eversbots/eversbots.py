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
from util import nearestPoint

#############
# FACTORIES #
#############

class EversbotAgents(AgentFactory):

  def __init__(self, isRed, first='offense', second='offense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

  def choose(self, agentStr, index):
    if agentStr == 'offense':
      return OffensiveReflexAgent(index)
    else:
      raise Exception("No agent identified by " + agentStr)

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
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

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


MZ_DISTS = {}
class ZoneAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        # Get OZone
        targets = self.getFood(gameState).asList()
        bounds = [0,0,100,100]
        mid = gameState.data.layout.height / 2
        for target in targets:
            if target[0] > bounds[0]:
                bounds[0] = target[0]
            if target[0] < bounds[2]:
                bounds[2] = target[0]
            if target[1] > bounds[1]:
                bounds[1] = target[1]
            if target[1] < bounds[3]:
                bounds[3] = target[1]
        candidates = util.Counter()
        for x in range(bounds[2] + 1, bounds[0]):
            for y in range(bounds[3] + 1, bounds[1]):
                if gameState.hasWall(x,y):
                    continue
                for target in targets:
                    dist = self.getMZDist((x,y),target)
                    if (self.index < 2 and target[1] > mid) or (self.index > 2 and target[1] < mid):
                        candidates[ (x,y) ] += dist * 5
                    else:
                        candidates[ (x,y) ] -= dist
        w,h = min(candidates.keys(),key=lambda y: candidates[y])
        def zone(*args):
            return (w,h)
        self.getOZone = zone

        # Prepare fields
        self.missingFood = []
        self.lastFood = self.getFoodYouAreDefending(gameState).asList()
        self.tailing = False
        self.foodCounter = 0

    def getMHDist(self,p1,p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
    def getMZDist(self,p1,p2):
        if MZ_DISTS.has_key((p1,p2)):
            return MZ_DISTS[(p1,p2)]
        else:
            dist = self.getMazeDistance(p1,p2)
            MZ_DISTS[(p1,p2)] = dist
            return dist

    def getFoodDist(self,gameState):
        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        zone = self.getOZone(gameState)
        return min([self.getMZDist(myPos,food) + self.getMZDist(zone,food) for food in foodList])

    def nextFoodCount(self):
        self.foodCounter += 1
        return self.foodCounter

    def getMissingFood(self, gameState):
        currentFood = self.getFoodYouAreDefending(gameState).asList()
        missing = [(self.nextFoodCount(),loc) for loc in self.lastFood if not loc in currentFood]
        response = []
        for i in range(len(self.missingFood)-1,-1,-1):
            response.append(self.missingFood[i][1])
            missing.append(self.missingFood[i])
        guess = util.Counter()
        for food in response:
            guess[food] = .5
        guess[self.getOZone()] = 1
        #self.display.updateDistributions([guess])
        self.lastFood = currentFood
        missing.sort()
        missing.reverse()
        missing = [missing[i] for i in range(min(len(missing),2))]
        self.missingFood = missing
        return response

    def getOpponentDists(self, gameState):
        dists = gameState.getAgentDistances()
        return [dists[i] for i in self.getOpponents(gameState)]

class OffensiveReflexAgent(ZoneAgent):
  """
  A reflex agent that seeks food. This is an agent
  is now updated, so let's hope it has what it takes
  """

  def getFeatures(self, gameState, action):
    # Preperation
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    # SuccessorScore
    features['successorScore'] = self.getScore(successor)

    # DistanceToFood
    if self.getFood(gameState).asList().count(myPos) == 0:
        features['distanceToFood'] = self.getFoodDist(successor)

    # InvaderDistance
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    pacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0 and myState.isPacman:
      dists = [self.getMZDist(myPos, a.getPosition()) for a in ghosts]
      features['ghostDist'] = min(dists)**.3
    if len(pacmen) > 0 and not myState.isPacman:
      dists = [self.getMZDist(myPos, a.getPosition()) for a in pacmen]
      features['pacDist'] = min(dists)
      if [a.getPosition() for a in pacmen].count(myPos) == 1:
        features['pacDist'] = 100

    # Def
    #missing = self.getMissingFood(gameState)
    #if len(missing) > 0 and sum(self.getOpponentDists(successor)) < 20 and not myState.isPacman:
    #    features['def'] = sum([self.getMZDist(myPos,food) for food in missing]) / float(len(missing))
    #    features['distanceToFood'] = 0 
    return features

  def getWeights(self, gameState, action):
    return { 'successorScore': 100, 'distanceToFood': -1, 'ghostDist': 8, 'def': -3, 'pacDist': -4}

