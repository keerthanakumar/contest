
import random
from captureAgents import CaptureAgent
from BaselineAgents.baselineAgents import OffensiveReflexAgent
from game import Directions
import util

#our code
from myUtil import *
from agent import MSBAgentRole

class MSBOffensiveAgentRole(MSBAgentRole):
	
	def getWeights(self):
		return self.factory.args["offenseWeights"]

	def getPositionRisk(self, gameState, pos, i = 0):
		
		total = 0

		# if the position is on our side, then there's 0 risk
		if gameState.isRed(pos) == self.red:
			return 0

		for enemyIndex in self.agent.getOpponents(gameState):
			enemyState = gameState.getAgentState(enemyIndex)
			beliefs = self.factory.getBeliefDistribution(enemyIndex)
			if enemyState.isPacman or enemyState.scaredTimer > i+1:
				continue
			if gameState.getAgentPosition(enemyIndex) == pos:
				total += 1000
				continue
			for enemyPos, prob in beliefs.items():
				total += prob * inv(self.distancer.getDistance(pos, enemyPos))

		return total

	def getPathRisk(self, gameState, source, target):
		return {pos: self.getPositionRisk(gameState, pos) for i, pos in enumerate(self.factory.getPath(gameState, source, target))}

	# returns tuple (foodPos, pathRisks)
	def getFoodToSeek(self, gameState):

		pos = gameState.getAgentPosition(self.index)
		allFood = self.agent.getFood(gameState).asList()

		# if any of the food is unreachable, it will cause problems with our A* pathfinding search -- so don't even try.
		# I don't think any of the contest maps have this, but I made a test map that does so that I could observe defense indefinitely without offense winning.
		allFood = filter(lambda f: self.distancer.getDistance(pos, f) < 1000, allFood)

		# bandaid fix for perf issues on large maps -- if >15 enemy food, just look at the 15 closest ones
		# in the future, maybe be smarter about this?
		maxNFood = self.factory.args["maxFoodToPathfind"]
		if len(allFood) > maxNFood:
			allFood.sort(key=lambda f: self.distancer.getDistance(pos, f))
			allFood = allFood[:maxNFood]

		choices = {food: self.getPathRisk(gameState, pos, food) for food in allFood}
		
		bestChoice = min(choices, key=lambda c: sum(choices[c].values())) #prev len(choices[c])
		return (bestChoice, choices[bestChoice])

	def computationsForFeatures(self, gameState):
		
		pos = gameState.getAgentPosition(self.index)
		enemyPositions = {enemyIndex: self.factory.getAveragedEnemyLocation(enemyIndex) for enemyIndex in self.agent.getOpponents(gameState)}
		foodGoal, foodPathRisks = self.getFoodToSeek(gameState)

		# draw path risks to the food we're seeking
		foodDistrib = util.Counter(foodPathRisks).copy()
		for f, prob in foodDistrib.items():
			if prob == 0:
				foodDistrib[f] += 0.001
		foodDistrib.normalize()
		self.agent.miscDistribution = foodDistrib

		# get nearest capsule
		capsules = self.agent.getCapsules(gameState)
		closestCapsule = min(capsules, key=lambda c: self.distancer.getDistance(c, pos)) if len(capsules)!=0 else None
		capsulePath = self.factory.getPath(gameState, pos, closestCapsule) if closestCapsule != None else []

		return {
			"foodGoal" : foodGoal,
			"enemyPositions" : enemyPositions,
			"closestCapsule" : closestCapsule,
			"capsulePath" : capsulePath
		}

	def getFeatures(self, gameState, action, successor, precomputed):
		
		features = {}

		pos = successor.getAgentPosition(self.index)
		onOwnTurf = gameState.isRed(pos) == self.red

		# feature to seek the single designated food calculated by getFoodToSeek
		features["foodDistance"] = inv(self.distancer.getDistance(pos, precomputed["foodGoal"]))

		# feature to avoid nearby ghosts, unless scared
		enemyDist = float("inf")
		closestEnemy = None
		for enemyIndex in precomputed["enemyPositions"]:
			enemyPos = gameState.getAgentPosition(enemyIndex)
			enemyState = gameState.getAgentState(enemyIndex)
			if enemyPos == None or enemyState.isPacman:
				continue
			dist = self.distancer.getDistance(pos, enemyPos)
			if enemyState.scaredTimer > dist+1:
				continue
			if dist < enemyDist:
				enemyDist = dist
				closestEnemy = enemyPos
		if enemyDist < float("inf"):
			features["closestGhost"] = enemyDist

		# if ghost is nearby, and capsule is nearby, try to get capsule
		closestCapsule = precomputed["closestCapsule"]
		capsulePath = precomputed["capsulePath"]
		if closestEnemy != None and closestCapsule != None:
			if closestEnemy not in capsulePath:
				features["capsuleDistance"] = inv(self.distancer.getDistance(pos, closestCapsule))

		# feature to avoid going in a dead end if a ghost is closely in tow
		if enemyDist < 3:
			if hasNWalls(pos, self.factory.walls, n=3):
				features["deathtrap"] = 1

		# feature to avoid thrashing
		if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction] or action == Directions.STOP:
			features["thrash"] = 1

		# feature to avoid getting stuck
		if action == Directions.STOP:
			stops = sum([1 if a == Directions.STOP else 0 for a in self.agent.moveHistory.list])
			if stops >= 3:
				features["deadlock"] = 1

		# feature to track invaders while in own territory if closer to us than other agent
		allyPos = None
		for allyIndex in self.factory.agents:
			if allyIndex == self.index:
				continue
			thisAllyPos = gameState.getAgentPosition(allyIndex)
			if allyPos == None or self.distancer.getDistance(pos, thisAllyPos) < self.distancer.getDistance(pos, allyPos):
				allyPos = thisAllyPos
		if allyPos != None and gameState.isRed(pos) == self.red:
			for enemyIndex in self.agent.getOpponents(gameState):
				if gameState.getAgentState(enemyIndex).isPacman:
					enemyPos = self.factory.getAveragedEnemyLocation(enemyIndex)
					if self.distancer.getDistance(pos, enemyPos) < self.distancer.getDistance(allyPos, enemyPos):
						features["closestInvader"] = inv(self.distancer.getDistance(pos, enemyPos))

		return features

