
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import game

#our code
from agent import *
from offenseRole import *
from defenseRole import *
from inference import *
import search
from myUtil import *

class MSBFactory(AgentFactory):

	def __init__(self, isRed, **kwArgs):
		AgentFactory.__init__(self, isRed)
		print "factory __init__ called"

		#for emergency timeout prevention
		self.totalRuntime = 0
		self.nRuntimeSamples = 0
		self.emergencyScale = 1

		self.args = kwArgs
		self.agents = {}
		
		#this lists which MSBAgentRoles we'll assign to new agents
		self.availableRoles = util.Queue()
		if not self.args["offenseOnly"]:
			self.availableRoles.push(MSBDefensiveAgentRole)
		self.availableRoles.push(MSBOffensiveAgentRole)
		
		self.initializeForNewGame()
	
	# this is (annoyingly) separate from __init__ because the factory outlasts a single game when the -n option is used
	def initializeForNewGame(self):

		print "initializeForNewGame called"

		self.distancer = None
		self.pathCache = {}
		self.walls = []
		self.legalPositions = []
		self.tracking = None
		self.gameTime = -1
		self.lastDeath = -1
		self.lastSwap = -1

	def getAgent(self, index):
		
		print "factory.getAgent called for agent %d" % index

		newAgent = MSBAgent(index, self)
		
		# assign this new agent a role
		if not self.args["doNothing"]:
			newAgentRole = self.availableRoles.pop()
			self.availableRoles.push(newAgentRole)
			newAgent.role = newAgentRole()

		self.agents[index] = newAgent
		return newAgent

	def removeDeadAgents(self, gameState):
		dead = []
		for agentIndex in self.agents:
			if agentIndex >= gameState.getNumAgents():
				dead.append(agentIndex)
		for agentIndex in dead:
			del self.agents[agentIndex]

	def initTracking(self, index, gameState):
		"Initializes inference modules for each enemy agent"

		#already inited
		if self.tracking is not None:
			return

		agent = self.agents[index]
		opponents = agent.getOpponents(gameState)

		self.tracking = {}

		for opp in opponents:
			tracker = ExactInference(MSBSimulatedEnemyAgent(opp, self.getDistancer(gameState), self.args["uniformEnemySimulation"]))
			tracker.failOnEmpty = self.args["failOnEmptyDistribution"]
			tracker.initialize(gameState)
			tracker.initializeSpecific(gameState.getInitialAgentPosition(opp))
			self.tracking[opp] = tracker

	def notifyEaten(self, gameState, enemyIndex):
		tracker = self.tracking[enemyIndex]
		tracker.initializeSpecific(gameState.getInitialAgentPosition(enemyIndex))

	def maybeSwapRoles(self, index):

		# if we don't have exactly 2 agents or we swapped recently, don't swap
		if len(self.agents) != 2 or self.gameTime - self.lastSwap < 10:
			return
		
		gameState = self.agents[index].getCurrentObservation()
		pos1, pos2 = [gameState.getAgentPosition(i) for i in self.agents]

		# if either agent is pacman, don't swap
		for agentIndex in self.agents:
			if gameState.getAgentState(agentIndex).isPacman:
				return

		# if the agents are far apart, don't swap
		if self.getDistancer(gameState).getDistance(pos1, pos2) > 7:
			return

		optFunc = max if self.agents[index].red else min
		closestXToBorder = optFunc(pos1, pos2, key=lambda p: p[0])[0]

		# if neither agent is close to the border, don't swap
		if abs(gameState.data.layout.width/2 - closestXToBorder) > 3:
			return

		# TODO: verify one of our agents died recently... not sure if necessary?

		for enemyIndex in self.tracking:
			
			# if there is an invader, don't swap
			if gameState.getAgentState(enemyIndex).isPacman:
				return

			# if there is a very imminent threat on the border, don't swap
			enemyPos = gameState.getAgentPosition(enemyIndex)
			if enemyPos != None and abs(gameState.data.layout.width/2 - enemyPos[0]) < 3:
				return

		# finally, if we haven't fallen out on any of the above conditions, swap roles
		self.lastSwap = self.gameTime
		roles = [agent.role for agent in self.agents.values()]
		for agent, newRole in zip(self.agents.values(), reversed(roles)):
			agent.role = newRole
		print "/\\"*100
		print "Swapping Roles!"


	def updateSharedKnowledge(self, index):

		#ideally, this shouldn't happen since we'll call initTracking from registerInitialState
		if self.tracking == None:
			self.initTracking(index, gameState)

		gameState = self.agents[index].getCurrentObservation()
		lastGameState = self.agents[index].getPreviousObservation()
		dists = gameState.getAgentDistances()

		#an annoyance
		self.removeDeadAgents(gameState)

		#update a timer to know how many moves have occurred before this one -- this includes enemy agents
		if self.gameTime == -1:
			startFirst = index == 0 #TODO this is incorrect, but it may not be possible to be truly correct.
			self.gameTime = 0 if startFirst else 1
		else:
			self.gameTime += 2
		print "Agent %d calling updateSharedKnowledge at time step %d" % (index, self.gameTime)

		#check if we died in the last step
		if lastGameState != None:
			lastPos = lastGameState.getAgentPosition(index)
			nowPos = gameState.getAgentPosition(index)
			if nowPos == gameState.getInitialAgentPosition(index) and self.getDistancer(gameState).getDistance(lastPos, nowPos) > 4:
				self.lastDeath = self.gameTime

		#check if the last enemy to move killed itself
		prevEnemy = index - 1
		prevEnemyEaten = False
		if prevEnemy < 0:
			prevEnemy += gameState.getNumAgents()
		prevAlly = prevEnemy - 1
		if prevAlly < 0:
			prevAlly += gameState.getNumAgents()
		if self.gameTime >= 2:
			prevAllyState = self.agents[prevAlly].getCurrentObservation()
			prevEnemyLoc = prevAllyState.getAgentPosition(prevEnemy)
			if prevEnemyLoc != None and (gameState.getAgentPosition(prevEnemy) == None or gameState.getAgentPosition(prevEnemy) == gameState.getInitialAgentPosition(prevEnemy)):
				for agentIndex in self.agents:
					if self.getDistancer(gameState).getDistance(prevEnemyLoc, prevAllyState.getAgentPosition(agentIndex)) <= 1:
						prevEnemyEaten = True
						break

		# check if an enemy ate one of our food in the last time step -- if so, we know where it is.
		prevEnemyJustAteFood = False
		prevEnemyFoodEaten = None
		if self.gameTime >= 2 and self.args["foodInference"]:
			prevAllyState = self.agents[prevAlly].getCurrentObservation()
			prevFood = set(self.agents[index].getFoodYouAreDefending(prevAllyState).asList())
			foodNow = set(self.agents[index].getFoodYouAreDefending(gameState).asList())
			foodDiff = prevFood - foodNow
			if len(foodDiff) == 1:
				prevEnemyJustAteFood = True
				prevEnemyFoodEaten = foodDiff.pop()

		for enemyIndex, tracker in self.tracking.items():
			print "Agent %d observes enemy %d at noisyDistance %d (direct reading: %s) from its viewpoint %s" % (index, enemyIndex, dists[enemyIndex], gameState.getAgentPosition(enemyIndex), gameState.getAgentPosition(index))

			# if the enemy is close enough for us to know exactly where it is, just update the tracker with that
			if gameState.getAgentPosition(enemyIndex) != None:
				tracker.initializeSpecific(gameState.getAgentPosition(enemyIndex))
				print "- it's close enough we have an exact reading, so ignoring noisyDistance"
				continue

			# if our check outside the loop indicated the enemy ate, skip observe and elapseTime on it
			if enemyIndex == prevEnemy and prevEnemyJustAteFood:
				tracker.initializeSpecific(prevEnemyFoodEaten)
				print "- enemy just ate food, so we know it's at %s" % tuple([prevEnemyFoodEaten])
				continue

			# if this enemy was the last enemy to move and killed itself, update beliefs to initial position
			if enemyIndex == prevEnemy and prevEnemyEaten:
				tracker.initializeSpecific(gameState.getInitialAgentPosition(enemyIndex))
				print "- enemy killed itself, resetting to initial position"
				continue

			# elapse time once per round
			if enemyIndex == prevEnemy and self.gameTime != 0:
				tracker.elapseTime(gameState.deepCopy())

			#debug
			# realPos = gameState.true.getAgentPosition(enemyIndex)
			# realDistance = util.manhattanDistance(gameState.getAgentPosition(index), realPos)
			# print "!!! agent %d's view of enemy %d: noisyDistance=%d, realPos=%s, realDistance=%d (delta %d)" % (index, enemyIndex, dists[enemyIndex], realPos, realDistance, dists[enemyIndex]-realDistance)
			# import capture
			# assert dists[enemyIndex]-realDistance in capture.SONAR_NOISE_VALUES, "invalid noisyDistance!!!"

			# observe
			tracker.observe(dists[enemyIndex], gameState.getAgentPosition(index), self.getDistancer(gameState), gameState)

			# if the enemy isPacman, then we know it's on our side. If not, we know it's not.
			if self.args["pacmanInference"]:
				usRed = self.agents[index].red
				isPacman = gameState.getAgentState(enemyIndex).isPacman
				locRed = isPacman if usRed else not isPacman
				beliefs = tracker.beliefs
				for loc in beliefs:
					if gameState.isRed(loc) != locRed:
						beliefs[loc] = 0
				beliefs.normalize()

			# not sure if I should need to do this, but the belief distribution seems to eventually be empty
			if tracker.getBeliefDistribution().totalCount() == 0:
				tracker.initializeUniformly(gameState)
				#observe again so distribution isn't useless
				tracker.observe(dists[enemyIndex], gameState.getAgentPosition(index), self.getDistancer(gameState), gameState)
				print "- enemy %d's tracker being reset due to being empty." % enemyIndex

			print "- enemy %d now thought to occupy %s" % (enemyIndex,self.getAveragedEnemyLocation(enemyIndex))

		self.maybeSwapRoles(index)

	def updateDisplay(self, gameState, curIndex):
		dists = [self.tracking[i].getBeliefDistribution() if i in self.tracking else None for i in range(gameState.getNumAgents())]
		if self.args["showMiscDistributions"]:
			for i in range(len(dists)):
				if dists[i] == None and i in self.agents:
					dists[i] = self.agents[i].miscDistribution
		self.agents[curIndex].displayDistributionsOverPositions(dists)

	def getBeliefDistribution(self, enemyIndex):
		return self.tracking[enemyIndex].getBeliefDistribution()

	def getAveragedEnemyLocation(self, enemyIndex):
		
		xavg = 0
		yavg = 0

		for pos, prob in self.tracking[enemyIndex].getBeliefDistribution().items():
			x, y = pos
			xavg += x * prob
			yavg += y * prob

		avgPoint = util.nearestPoint((xavg, yavg))

		# annoying thing because mazeDistance doesn't work if one point is a wall
		if avgPoint in self.walls:
			neighbors = list(getNeighbors(avgPoint))
			neighbors = [n for n in neighbors if n in self.legalPositions]
			if len(neighbors) > 0:
				avgPoint = neighbors[0]
			else:
				raise Exception("avg enemy location is wall surrounded by walls")

		return avgPoint

	def getDistancer(self, gameState = None):
		if self.distancer != None:
			return self.distancer

		# this should never happen, since registerInitialState calls this with a gameState
		if gameState == None:
			raise Exception("getDistancer called without gameState, but no distancer has been inited yet")

		distancer = distanceCalculator.Distancer(gameState.data.layout)
		distancer.getMazeDistances()
		self.distancer = distancer
		self.walls = gameState.getWalls().asList()
		self.legalPositions = gameState.getWalls().asList(False)
		return distancer

	def getPath(self, gameState, source, target):
		
		# basic caching of paths
		if (source, target) in self.pathCache:
			#print "Found path from %s to %s in pathCache" % (source, target)
			return self.pathCache[(source, target)]
		elif (target, source) in self.pathCache:
			#print "Found path from %s to %s in pathCache" % (source, target)
			return reversed(self.pathCache[(target, source)])

		print "getPath(%s, %s) called, computing using A*" % (source, target)
		# compute path using A* search with known optimal maze distance as heuristic
		problem = MSBPathfindingSearchProblem(source, target, self.legalPositions)
		def heuristic(state, prob):
			return self.getDistancer(gameState).getDistance(state, target)
		path = search.astar(problem, heuristic)
		assert len(path) == self.getDistancer(gameState).getDistance(source, target), "A* found non-optimal path from %s to %s" % (source, target)
		
		# update cache
		self.pathCache[(source, target)] = path
		for i in range(0,len(path)-1):
			self.pathCache[(path[i], target)] = path[i+1:]

		print "getPath(%s, %s) returning; len(pathCache)=%d" % (source, target, len(self.pathCache))
		return path

	def reportRuntime(self, elapsedTime):
		self.totalRuntime += elapsedTime
		self.nRuntimeSamples += 1
		avgRuntime = self.totalRuntime / self.nRuntimeSamples
		if avgRuntime > 0.7 and self.emergencyScale >= 0.4:
			self.emergencyScale -= 0.25
			if "original_maxFoodToPathfind" not in self.args:
				self.args["original_maxFoodToPathfind"] = self.args["maxFoodToPathfind"]
			self.args["maxFoodToPathfind"] = int(self.args["original_maxFoodToPathfind"] * self.emergencyScale)
			print "########################### Emergency timeout prevention: reducing maxFoodToPathfind to %d (last move took %.3f seconds; average is %.3f seconds)" % (self.args["maxFoodToPathfind"], elapsedTime, avgRuntime)
			self.totalRuntime = 0
			self.nRuntimeSamples = 0

# this is used to come up with a distribution of possible enemy agent successor positions for the elapseTime updates
# TODO: maybe in the future, use a copy of our own agent to predict this?
class MSBSimulatedEnemyAgent:

	def __init__(self, index, distancer, uniform = False):
		self.index = index
		self.distancer = distancer
		self.weights = {
			"default" : 1
		}

		if uniform:
			self.agent = self
			return

		#currently, this uses BaselineAgents' agents.
		#To use the features/weights in this class, set self.agent = self
		try:
			from BaselineAgents.baselineAgents import OffensiveReflexAgent, DefensiveReflexAgent
			self.agent = OffensiveReflexAgent(index) if index%2==0 else DefensiveReflexAgent(index)
			self.agent.distancer = distancer
		except: #if BaselineAgents isn't accessible, fallback gracefully
			self.agent = self

	def getFeatures(self, state, action):
		return {"default":1}

	def evaluate(self, state, action):
		
		#fall through to another agent if we have one
		if self.agent is None:
			return 0
		elif self.agent != self:
			return max(0, self.agent.evaluate(state, action))

		features = self.getFeatures(state, action)
		amts = [features[i]*self.weights[i] if i in self.weights else 0 for i in features]
		return max(sum(amts), 0)

	def getDistribution(self, gameState):
		
		#get the utilities from the agent and find the maximum
		utilities = {action: self.evaluate(gameState, action) for action in gameState.getLegalActions(self.index)}
		maxUtility = max(utilities.values())
		
		#any action that maximizes utility gets equal probability, all else get 0
		tbr = util.Counter()
		for action in utilities:
			if utilities[action] < maxUtility:
				continue
			tbr[action] = 1
		tbr.normalize()

		return tbr

class MSBPathfindingSearchProblem(search.SearchProblem):
	
	def __init__(self, source, target, legalPositions):
		self.source = source
		self.target = target
		self.legalPositions = legalPositions

	def getStartState(self):
		return self.source
	
	def isGoalState(self, state):
		return state == self.target
	
	def getSuccessors(self, state):
		return [(c, c, 1) for c in getNeighbors(state, includeDiagonals=False) if c in self.legalPositions]
	
	def getCostOfActions(self, actions):
		return len(actions)