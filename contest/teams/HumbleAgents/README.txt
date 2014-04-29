Jaime Rivera rjr2426
Keerthana Kumar kk24268

Although, the code is messy, this should hopefully be organized.

Team Make-Up:   1 Defense 1 Offense
Agent Type:     Reflex
Inference:      Exact

Defensive Agent:
----------------

The Defensive Agent employs three different strategies:
- Pellet Path Prediction
- Offensive Tracking
- Full Retreat

Pellet Path Prediction:

- If we don't know their Team Make-Up, we simply assume
that they're going to go for the closest piece on our grid.

- We will allow them to take our pellets in order to determine
who is offensive on their team.

Offensive Tracking:

- If we know that some, but not all are on offense. We will minimize
our distance as a defensive agent to their tracked location.

Full Retreat:

- If all of their agents are on the offensive, we shall assume our
optimal defensive location.

- If there exists an area of pellets thathas only one entrance and
the number of pellets in that area is at least 3, we shall find such
closest entrance and place ourselves at this location. Since no contest
layout has a capsule, this guarantees a win if their agent doesn't switch
back to defense. This calculation is naively calculated and takes
advantage of all contest layouts.

- If there doesn't exist such an entrance, then we will minimize our squared
distance to all pieces and prioritize on scaring off ghosts while staying
centered on our pieces.


Offensive Agent:
----------------

The Offensive Agent is just gutsy. It does what it wishes.

The Offensive Agent employs three strategies:
- Cautious Baiting
- Fast Retrieval

Cautious Baiting:

- If there is an enemy that can attack this agent, it shall try
its best to become a pacman and move into locations that gaurantee it would
not die on its next turn. If it's inevitable, then it shall suicide. Otherwise,
it will take the next opening it can. When a fork is reached, it shall depth first
search for a depth of 13 discovering the number of open locations and number of
pellets found. It prioritizes on a function of these two values. This helps eliminate
some corner problems found with simply using minimum distance; however, min distance
to food is still used.

- If the defensive agent is on full retreat, meaning all agents are on the offensive,
it shall use solely a TSP approximation algorithm using MST's to find a decent tour of
all pellets within the grid. This is to bound the worst case of simply finding the minimum
distanced pellet.

Inference:
----------

Standard Exact Inference is used - very similar to our tracking project.

Some clever steps are made though:

- Using the Agent State of our enemies, we can elminate any
beliefs on our side of the board if they're not a pacman and vice versa. 

- Also, for each Teammate we have, we can elminate any beliefs that would
be in the vision of our Teammate.

- Furthermore, we determine which pellets have been eaten to determine the
most likely enemy to be at the location. If we assume incorrectly, this
is easily updated and fixed at the next step.

A problem: We decided to not bother about establishing the location of the enemy
agent after we have eaten them. This was too hairy, and we noticed that a few steps
of our inference accurately detected the position of the agent afterwards - so this
issue was ignored.


Hopefully, we do well :)
