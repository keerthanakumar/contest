
import traceback
import time
import random
from captureAgents import CaptureAgent
from BaselineAgents.baselineAgents import OffensiveReflexAgent
from game import Directions
import util

#our code
from myUtil import *

class MSBAgent(CaptureAgent):

	def __init__(self, index, factory, role = None):
		CaptureAgent.__init__(self, index)
		self.factory = factory
		self.role = role
		self.miscDistribution = None
		self.moveHistory = util.Queue()

	# inspired by Josh's dontPanic decorator -- thanks for the inspiration!
	def getAction(self, gameState):
		
		startTime = time.time()
		if self.factory.args["noDying"]:
			try:
				action = CaptureAgent.getAction(self, gameState)
			except:
				print "exception caught by noDying, choosing random action"
				print traceback.format_exc()
				action = random.choice(gameState.getLegalActions(self.index))
		else:
			action = CaptureAgent.getAction(self, gameState)
		endTime = time.time()
		
		# update moveHistory with this move
		self.moveHistory.push(action)
		if len(self.moveHistory.list) > 3:
			self.moveHistory.pop()

		print "Agent %d using %s returned action %s in %.3f seconds" % (self.index, self.role, action, endTime-startTime)
		self.factory.reportRuntime(endTime-startTime)
		return action

	def chooseAction(self, gameState):

		# gameState will already be in self.observationHistory so the factory can see/use it
		self.factory.updateSharedKnowledge(self.index)

		if self.role == None:
			return Directions.STOP

		# update role properties, in case role is new
		self.role.factory = self.factory
		self.role.index = self.index
		self.role.distancer = self.factory.getDistancer(gameState)
		self.role.red = self.red
		self.role.agent = self

		# ask role what action to take
		action = self.role.getAction(gameState)

		# check if we're going to eat anything
		for enemyIndex in self.getOpponents(gameState):
			pos = gameState.getAgentPosition(enemyIndex)
			if pos == None:
				continue
			dist = self.factory.getDistancer(gameState).getDistance(pos, gameState.getAgentPosition(self.index))
			if dist > 1:
				continue
			successor = gameState.generateSuccessor(self.index, action)
			if successor.getAgentPosition(enemyIndex) == None or successor.getAgentPosition(enemyIndex) == successor.getInitialAgentPosition(enemyIndex):
				self.factory.notifyEaten(gameState, enemyIndex)

		self.factory.updateDisplay(gameState, self.index)
		return action

	def registerInitialState(self, gameState):
		"Implementation adapted from captureAgents.py to share a distancer for the whole team"

		print "registerInitialState called for agent %d" % self.index

		#an annoying thing... factory and agent objects persist between games; if we're the first agent from our team to be initialized, tell the factory it's a new game.
		if self.index <= 1:
			self.factory.initializeForNewGame()

		self.red = gameState.isOnRedTeam(self.index)
		self.distancer = self.factory.getDistancer(gameState)
		self.moveHistory = util.Queue()

		#if we're the first agent from the team to run, tell the factory to initialize the inference modules
		self.factory.initTracking(self.index, gameState)

		import __main__
		if '_display' in dir(__main__):
			self.display = __main__._display

class MSBAgentRole:

	# these properties are set by MSBAgent
	factory = None
	index = None
	distancer = None
	red = None
	agent = None

	# set by getAction if not already set
	weights = None

	def getFeatures(self, gameState, action, successor, precomputed):
		assert False # abstract

	def getWeights(self):
		assert False # abstract

	def computationsForFeatures(self, gameState):
		return None

	def getAction(self, gameState):

		if self.weights == None:
			self.weights = util.Counter(self.getWeights())

		precomputed = self.computationsForFeatures(gameState)
		def optionTuple(action):
			features = self.getFeatures(gameState, action, gameState.generateSuccessor(self.index, action), precomputed)
			score = self.weights * features
			return (action, features, score)
		options = [optionTuple(action) for action in gameState.getLegalActions(self.index)]
		
		printDebugFeatureTable(self.weights, options)

		bestUtility = max(options, key=lambda x: x[2])[2]
		bestActions = [option[0] for option in options if option[2] >= bestUtility]

		return random.choice(bestActions)

	def __str__(self):
		return self.__class__.__name__
