
import random
from captureAgents import CaptureAgent
from BaselineAgents.baselineAgents import OffensiveReflexAgent
from game import Directions
import util

#our code
from myUtil import *
from agent import MSBAgentRole

class MSBDefensiveAgentRole(MSBAgentRole):
	
	def getWeights(self):
		return self.factory.args["defenseWeights"]

	def computationsForFeatures(self, gameState):

		# determine which positions are on the border
		legalPositions = self.factory.legalPositions
		borderX = int(gameState.data.layout.width/2) - 1 if self.red else int(gameState.data.layout.width/2)
		borderRadius = [0] #[0,1] if self.red else [0,-1]
		borderPositions = [pos for pos in legalPositions if borderX-pos[0] in borderRadius]

		# get positions of enemies and detect whether they're invaders
		enemyPositions = {enemyIndex: self.factory.getAveragedEnemyLocation(enemyIndex) for enemyIndex in self.agent.getOpponents(gameState)}
		invaders = [enemyIndex for enemyIndex in enemyPositions if gameState.getAgentState(enemyIndex).isPacman]

		# calculate a distribution of the likelihood of each border position being an entry point
		borderDistrib = util.Counter()
		for pos in borderPositions:
			for enemyIndex in self.agent.getOpponents(gameState):
				if enemyIndex in invaders:
					continue
				borderDistrib[pos] += 2 ** -self.distancer.getDistance(pos, enemyPositions[enemyIndex])
		borderDistrib.normalize()
		
		# detect end game priority locations (3+ food surrounded by walls; can keep from losing by sitting in front of it)
		def guardLoc(clump):
			spaces = set()
			for pos in clump:
				for pn in getNeighbors(pos, includeDiagonals=False):
					if pn in legalPositions and pn not in clump:
						spaces.add(pn)
			return spaces.pop() if len(spaces) == 1 else None
		food = self.agent.getFoodYouAreDefending(gameState).asList()
		clumps = filter(lambda l: len(l)>=3, getFoodClumps(food))
		guardLocs = filter(lambda x: x != None, map(guardLoc, clumps))

		# paint the borderDistrib and guardLocs for this agent's miscDistribution
		self.agent.miscDistribution = borderDistrib.copy()
		for gl in guardLocs:
			self.agent.miscDistribution[gl] += 0.1
		self.agent.miscDistribution.normalize()

		# pick a location near the border with minimal total distance to the border
		borderGuardCol = borderX + (-2 if self.red else 2)
		borderGuardColSpots = [p for p in legalPositions if p[0] == borderGuardCol]
		borderGuardColBest = min(borderGuardColSpots, key=lambda p: sum([self.distancer.getDistance(p, bp) for bp in borderPositions]))

		return {
			"borderDistrib" : borderDistrib,
			"borderPositions" : borderPositions,
			"enemyPositions" : enemyPositions,
			"invaders" : invaders,
			"endGameGuardLocs" : guardLocs,
			"food" : food,
			"borderGuardLocation" : borderGuardColBest
		}

	def getFeatures(self, gameState, action, successor, precomputed):
		
		features = {}
		pos = successor.getAgentPosition(self.index)

		borderDistrib = precomputed["borderDistrib"]
		invaders = precomputed["invaders"]
		enemyPositions = precomputed["enemyPositions"]

		# calculate the weighted average distance to the border positions
		weightedDist = sum([prob * self.distancer.getDistance(pos, borderPos) for borderPos, prob in borderDistrib.items()])
		features["weightedBorderDistance"] = inv(weightedDist)

		# try to move toward the optimal border guard location
		"""
		features["borderGuardLocDist"] = inv(self.distancer.getDistance(pos, precomputed["borderGuardLocation"]))
		"""

		# add a feature to prevent us from going to the other side
		# if there's an invader, we ignore this and go for it at all costs
		if gameState.isRed(successor.getAgentPosition(self.index)) != self.red and len(invaders) == 0:
			features["inEnemyTerritory"] = 1

		# add a feature to minimize distance to closest invader
		if len(invaders) > 0:
			print "invader at pos %s has distance %d from pos %s (action %s from %s)" % (enemyPositions[invaders[0]], self.distancer.getDistance(pos, enemyPositions[invaders[0]]), pos, action, gameState.getAgentPosition(self.index))
			closestInvader = min([self.distancer.getDistance(enemyPositions[i], pos) for i in invaders])
			features["closestInvader"] = inv(closestInvader)

		# if we are near the end of the game and/or only have one trapped food clump, go defend it
		guardLocs = precomputed["endGameGuardLocs"]
		food = precomputed["food"]
		isEndGame = len(food) < 5
		isEndGameLooser = len(food) < 7
		if (isEndGame and len(guardLocs)>0) or (isEndGameLooser and len(guardLocs)==1):
			closestGuardLoc = min([self.distancer.getDistance(pos, gl) for gl in guardLocs])
			features["endGameGuardLoc"] = inv(closestGuardLoc)

		return features