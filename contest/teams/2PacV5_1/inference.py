
# adapted from inference.py implementation in PS4: Tracking
# irrelevant parts removed/ported
# individual (not joint) particle filters are used for each ghost

import itertools
import util
import random
import game

import sys
import cStringIO
from pprint import pprint
import inspect

class InferenceModule:
	"""
	An inference module tracks a belief distribution over a ghost's location.
	This is an abstract class, which you should not modify.
	"""

	############################################
	# Useful methods for all inference modules #
	############################################

	def __init__(self, ghostAgent):
		"Sets the ghost agent for later access"
		self.ghostAgent = ghostAgent
		self.index = ghostAgent.index
		self.obs = [] # most recent observation position

	def getPositionDistribution(self, gameState):
		"""
		Returns a distribution over successor positions of the ghost from the given gameState.

		You must first place the ghost in the gameState, using setGhostPosition below.
		"""
		ghostPosition = gameState.getAgentPosition(self.index) # The position you set
		actionDist = self.ghostAgent.getDistribution(gameState)
		dist = util.Counter()
		for action, prob in actionDist.items():
			successorPosition = game.Actions.getSuccessor(ghostPosition, action)
			dist[successorPosition] = prob
		return dist

	def setGhostPosition(self, gameState, ghostPosition):
		"""
		Sets the position of the ghost for this inference module to the specified
		position in the supplied gameState.

		Note that calling setGhostPosition does not change the position of the
		ghost in the GameState object used for tracking the true progression of
		the game.  The code in inference.py only ever receives a deep copy of the
		GameState object which is responsible for maintaining game state, not a
		reference to the original object.  Note also that the ghost distance
		observations are stored at the time the GameState object is created, so
		changing the position of the ghost will not affect the functioning of
		observeState.
		"""
		conf = game.Configuration(ghostPosition, game.Directions.STOP)
		gameState.data.agentStates[self.index] = game.AgentState(conf, False)
		return gameState

	def initialize(self, gameState):
		"Initializes beliefs to a uniform distribution over all positions."
		self.legalPositions = gameState.getWalls().asList(False)
		# previously: self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
		self.initializeUniformly(gameState)

	######################################
	# Methods that need to be overridden #
	######################################

	def initializeUniformly(self, gameState):
		"Sets the belief state to a uniform prior belief over all positions."
		pass

	def observe(self, observation, viewpointLocation, distancer, gameState):
		"Updates beliefs based on the given distance observation and gameState."

		"""
		I've changed the signature of this method a bit to allow for a shared inference module that incorporates observations from multiple team members
		"""
		pass

	def elapseTime(self, gameState):
		"Updates beliefs for a time step elapsing from a gameState."
		pass

	def getBeliefDistribution(self):
		"""
		Returns the agent's current belief state, a distribution over
		ghost locations conditioned on all evidence so far.
		"""
		pass

class ExactInference(InferenceModule):
	"""
	The exact dynamic inference module should use forward-algorithm
	updates to compute the exact belief function at each time step.
	"""

	# debug
	def assertNotZero(self, gameState, msg = None, prev = None):
		if msg == None:
			traceback = inspect.getframeinfo(inspect.currentframe().f_back)
			msg = "%s:%d" % (traceback.function, traceback.lineno)
		if not 0.999 <= self.beliefs.totalCount() <= 1.001 and prev != None:
			suppressed = " (suppressed)" if not self.failOnEmpty else ""
			print "assertion about to fire%s; previous valid belief distribution:" % suppressed
			#self.printBeliefs(gameState, prev)
		if self.failOnEmpty:
			assert 0.999 <= self.beliefs.totalCount() <= 1.001, "[enemy %d] invalid beliefs sum %.3f at %s" % (self.index, self.beliefs.totalCount(), msg)

	"""
	def printBeliefs(self, gameState, beliefs = None):
		
		# this function is apparently problematic sometimes even when cStringIO is not in use; to avoid errors, let's not use it
		if True:
			print "@@@ printBeliefs disabled for safety, re-enable in inference.py if needed"
			return

		# cStringIO output used with -Q does not support unicode
		# thus, calling this function would cause a crash, and the output isn't going anywhere anyway.
		if type(sys.stdout) == cStringIO.OutputType:
			return

		beliefs = self.beliefs if beliefs == None else beliefs
		chars = ["-", " ", "1", "2", "3", "4", "5", "6"]
		grid = gameState.getWalls().copy()
		rows = ""
		for y in range(grid.height-1,-1,-1):
			row = ""
			for x in range(grid.width):
				beliefChar = chars[1 + int((len(chars)-2)*beliefs[(x,y)])]
				grid[x][y] = chars[0] if grid[x][y] == True else beliefChar
				row += grid[x][y]
			rows += "\u2551 %s \u2551\n" % row
		pad = u"\u2550"*(grid.width+2)
		print u"\u2554%s\u2557\n%s\u2558%s\u255B" % (pad, rows, pad)
	"""

	def __init__(self, ghostAgent):
		InferenceModule.__init__(self, ghostAgent)
		self.failOnEmpty = True

	def initializeSpecific(self, loc):
		self.beliefs = util.Counter()
		self.beliefs[loc] = 1

	def initializeUniformly(self, gameState):
		"Begin with a uniform distribution over ghost positions."
		self.beliefs = util.Counter()
		for p in self.legalPositions: self.beliefs[p] = 1.0
		self.beliefs.normalize()

	def observe(self, observation, viewpointLocation, distancer, gameState):
		"""
		I've changed the signature of this method a bit to allow for a shared inference module that incorporates observations from multiple team members
		"""

		print "entering observe for enemy %d" % self.index

		self.assertNotZero(gameState)
		prev = self.beliefs.copy()

		noisyDistance = observation
		pacmanPosition = viewpointLocation

		allPossible = util.Counter()
		for p in self.legalPositions:
			trueDistance = util.manhattanDistance(pacmanPosition, p)
			#previously, had distancer.getDistance(pacmanPosition, p) but sonar uses manhattan distances
			allPossible[p] = gameState.getDistanceProb(trueDistance, noisyDistance)*self.beliefs[p]
			if allPossible[p] < 0.0001 and self.beliefs[p] > 0.0001:
				print "zeroed location %s, emission(%d, %d)=%.4f, previously %.4f" % (p, trueDistance, noisyDistance, gameState.getDistanceProb(trueDistance, noisyDistance), self.beliefs[p])

		allPossible.normalize()
		self.beliefs = allPossible

		self.assertNotZero(gameState, prev=prev)

	def elapseTime(self, gameState):

		self.assertNotZero(gameState)

		newBeliefs = util.Counter()
		for oldPos in self.legalPositions:

			newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

			for newPos, prob in newPosDist.items():
				assert newPos in self.legalPositions, "illegal newPos in elapseTime"
				newBeliefs[newPos] += self.beliefs[oldPos] * prob

		newBeliefs.normalize()
		self.beliefs = newBeliefs

		self.assertNotZero(gameState)

	def getBeliefDistribution(self):
		return self.beliefs