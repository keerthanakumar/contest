# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
from game import Directions
from util import nearestPoint
import game, time, random, util

#############
# FACTORIES #
#############

class KillerAgents(AgentFactory):

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    locator = AgentLocator()
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

###########
# Locator #
###########
class AgentLocator():
    
    def __init__(self):
        self.team = []
        self.opponents = []
        self.states = []
        self.beliefs = []
        self.red = False
        self.distancer = None

    def initializeState(self, gameState, distancer, index):
        if len(self.states) != 0:
            return
        self.states.append(gameState)
        self.red = gameState.isOnRedTeam(index)
        self.distancer = distancer

        for index in range(gameState.getNumAgents()):
            self.beliefs.append(None)
            if gameState.isOnReadTeam(index) == self.red:
                self.team.append(index)
            else:
                self.opponents.append(index)
        
        self.legalPositions = [p for p in gameState.getWalls().asList(False)]
        self.__initializeBeliefs()

    def __initializeBelief(self):
        for index in self.opponents:
            self.beliefs[index] = util.Counter()
            position = self.states[-1].getInitialAgentPosition(index)
            self.beliefs[index][position] = 1.0

##########
# Agents #
##########

class KillerAgent(CaptureAgent):

    def registerInitalState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.locator.initializeState(gameState, self.distancer, self.index)

    def initializeLocator(self, locator):
        self.locator = locator

    def getSuccesor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        self.locator.observe(gameState, self.index)
        self.chooseTarget(gameState)
        return self.bestTargetAction(gameState)

    def bestTargetAction(self, gameState):
        targetPos = self.targeter.getTarget(self.index)
        actionResults = util.Counter()
        actions = gameState.getLegalActions(self.index)
        actions.remove(Direction.STOP)
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            newPos = successor.getActionPosition(self.index)
            actionResults[action] = -self.getMazeDistance(newPos, targetPos)
        return actionResults.argMax()

    def chooseTarget(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = gameState.getAgentPosition(self.index)
        self.invaders = pi fp"
