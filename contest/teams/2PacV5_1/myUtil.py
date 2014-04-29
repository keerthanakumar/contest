
import sys, cStringIO

# gets neighboring tiles, optionally including diagonals. Does not check legality
def getNeighbors(pos, includeDiagonals = True):
	x, y = pos
	for dx in [-1,0,1]:
		for dy in [-1,0,1]:
			if dx == 0 and dy == 0:
				continue
			if dx != 0 and dy != 0 and not includeDiagonals:
				continue
			yield (x+dx, y+dy)

# returns true if the position has N walls or more surrounding it
def hasNWalls(pos, walls, n = 3):
	count = 0
	for p in getNeighbors(pos, includeDiagonals=False):
		if p in walls:
			count += 1
	return count >= n

# uses a simple union-find algorithm to find clumps of food
def getFoodClumps(food):
	
	# setup union-find table
	table = {f:f for f in food}
	def find(x):
		if table[x] == x:
			return x
		else:
			table[x] = find(table[x])
			return table[x]
	def union(x, y):
		xr = find(x)
		yr = find(y)
		table[yr] = xr

	# detect adjacent foods, union their sets together
	for f in food:
		for fn in getNeighbors(f, includeDiagonals=False):
			if fn in food:
				union(f, fn)

	# convert from union-find table to a list of lists
	clumpsByRoot = {}
	for f in table:
		fr = find(f)
		if fr not in clumpsByRoot:
			clumpsByRoot[fr] = []
		clumpsByRoot[fr].append(f)
	return sorted(clumpsByRoot.values(), key=lambda l: len(l), reverse=True)

#prevent div-by-0 errors
def inv(n):
	return 1.0/float(n) if n != 0 else 100

#prints a table for debugging including values of all features and weights
#each member of options is an (action, features, score) tuple
def printDebugFeatureTable(weights, options):

	# if running with -Q, don't bother trying to print this stuff.
	if type(sys.stdout) == cStringIO.OutputType:
		return

	gutter = " | "
	colWidths = [25, 15, 11, 11, 11, 11, 11]
	nCols = len(colWidths)
	tableWidth = sum(colWidths) + len(gutter)*(nCols-1)

	allActions = ["North", "South", "East", "West", "Stop"]
	actions = {}
	for option in options:
		actions[option[0]] = option

	#header row
	print "-"*tableWidth
	headers = ["Feature", "Weight"]+allActions
	print gutter.join(["%*s" % (-colWidths[i], s) for i, s in enumerate(headers)])
	print "-"*tableWidth

	#print rows for each features
	for feature in weights:
		if weights[feature] == 0:
			continue
		vals = [feature, weights[feature]]+[actions[action][1][feature] if action in actions and feature in actions[action][1] else "" for action in allActions]
		print gutter.join([("%*s" if type(v)==type('') else "%*f") % (colWidths[i], v) for i, v in enumerate(vals)])
	print gutter.join([" "*colWidths[i] for i in range(nCols)])
	totals = ["TOTAL", ""]+[actions[action][2] if action in actions else "" for action in allActions]
	print gutter.join([("%*s" if type(v)==type('') else "%*f") % (colWidths[i], v) for i, v in enumerate(totals)])
	print "-"*tableWidth