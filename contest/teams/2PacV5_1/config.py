# config.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import os

"""
-----------------------
  Agent Configuration
-----------------------

Settings:

 - TeamName (string)
    The official name of your team. Names
    must be alpha-numeric only. Agents with
    invalid team names will not execute.

 - AgentFactory (string)
    The fully qualified name of the agent
    factory to execute.

 - AgentArgs (dict of string:string)
    Arguments to pass to the agent factory

 - NotifyList (list of strings)
    A list of email addresses to notify
    to when this agent competes.

 - Partners (list of strings)
    Group members who have contributed to
    this agent code and design.

"""

# Alpha-Numeric only
TeamName = os.path.split(os.path.abspath(os.path.join(__file__, os.pardir)))[1]
# NOTE: this ugly thing just gets the folder name to be the team name. This should be something like "2PacV[n]" where [n] is a version number. --Matt
# Thanks StackOverflow! http://stackoverflow.com/questions/2860153/get-parent-directory-in-python

# Filename.FactoryClassName (CASE-sensitive)
AgentFactory = 'factory.MSBFactory'

Partners = ['Matt Broussard']

AgentArgs = {

    # True catches all exceptions that could arise from getAction and returns a random choice
    # This is to prevent the agent from ever crashing during the contest
    # We set this to automatically be True for contest submissions, whose folder names will contain "V[n]" versioning
    "noDying" : "V" in TeamName,

    # True enables showing of miscellaneous distributions (colored for our agents)
    # What this actually entails is context-sensitive.
    # One such distribution is the belief distribution over likely border crossing points for an enemy agent
    "showMiscDistributions" : True,

    # True causes enemy motion to be predicted based on equal probability of any action
    # False causes enemy motion to be predicted based on OffensiveReflexAgent
    "uniformEnemySimulation" : True,

    # True causes agent to sit one spot and not move (used for inference debugging)
    "doNothing" : False,

    # True allows us to update trackers when we detect the enemy has eaten food between previous and current time step -- tells us where they are.
    "foodInference" : True,

    # True allows us to update trackers when we detect the enemy crossing the border
    "pacmanInference" : True,

    # True causes asserts in inference.py to fail if the belief distribution ever becomes empty
    # False temporarily disables this behavior, instead just uniformly reinitializing when this happens
    "failOnEmptyDistribution" : False,

    # max number of enemy food to pathfind to at any given timestep -- this helps us stay under the time limit on big maps
    "maxFoodToPathfind" : 15,

    # True causes both agents to be offensive
    "offenseOnly" : False,

    # weights for the defensive agent role's features
    "defenseWeights" : {
        "weightedBorderDistance" : 1,
        "borderGuardLocDist" : 1,
        "inEnemyTerritory" : float("-inf"),
        "closestInvader" : 10000,
        "endGameGuardLoc" : 30000
    },

    # weights for the offensive agent role's features
    "offenseWeights" : {
        "foodDistance" : 10,
        #"closestFood1" : 10,
        #"closestFood2" : 3,
        #"closestFood3" : 1,
        "closestGhost" : 0, #15, # for some reason, this causes bad thrashing behavior
        "thrash" : 0, #-1000
        "deadlock" : -1000,
        "closestInvader" : 20,
        "capsuleDistance" : 50,
        "deathtrap" : -10000,
    },

}

NotifyList = ["mattb@cs.utexas.edu"]
