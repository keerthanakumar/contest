# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

# search.py implemented for assignment 1

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def genericSearch(problem, approach="bfs", heuristic=nullHeuristic):

    #if we start at the goal, do nothing
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []
    
    #setup frontier depending on approach
    frontier = None
    if approach == "dfs":
        frontier = util.Stack()
    elif approach == "bfs":
        frontier = util.Queue()
    elif approach == "ucs":
        frontier = util.PriorityQueueWithFunction(priorityFunction=lambda x: problem.getCostOfActions(x[0] + [x[1][1]]))
    elif approach == "astar":
        frontier = util.PriorityQueueWithFunction(priorityFunction=lambda x: heuristic(x[1][0], problem) + problem.getCostOfActions(x[0] + [x[1][1]]))
    else:
        return None

    #setup visited set, add initial successors to frontier
    visited = set()
    visited.add(start)
    map(lambda x: frontier.push(([], x)), problem.getSuccessors(start))

    #conduct search
    while not frontier.isEmpty():
        path, successor = frontier.pop()
        successor, action, stepCost = successor
        
        if successor in visited:
            continue
        else:
            visited.add(successor)

        if problem.isGoalState(successor):
            return path + [action]

        map(lambda x: frontier.push((path + [action], x)), problem.getSuccessors(successor))

    #if we get here, we didn't find anything
    return None

def depthFirstSearch(problem):
    return genericSearch(problem, approach="dfs")

def breadthFirstSearch(problem):
    return genericSearch(problem, approach="bfs")

def uniformCostSearch(problem):
    return genericSearch(problem, approach="ucs")

def aStarSearch(problem, heuristic=nullHeuristic):
    return genericSearch(problem, approach="astar", heuristic=heuristic)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
