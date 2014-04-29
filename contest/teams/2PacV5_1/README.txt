Name: Matt Broussard
 EID: msb2774
CSID: mattb
Date: 4/27/2014

PacMan didn't choose the thug life, the thug life chose PacMan...

    ▄████▄          ___   ____                     ▒▒▒▒▒ 
   ███▄█▀          |__ \ / __ \____ ______        ▒ ▄▒ ▄▒
  ▐████  █  █  █   __/ // /_/ / __ `/ ___/ █  █   ▒▒▒▒▒▒▒
   █████▄         / __// ____/ /_/ / /__          ▒▒▒▒▒▒▒
    ▀████▀       /____/_/    \__,_/\___/          ▒ ▒ ▒ ▒

My agent will be submitted multiple times with versioned names. Thus the first submission will be "2PacV1", followed by "2PacV2", etc.

*** Version 1 (submitted 4/24/2014):

Right now both agents are offensive and pretty simple. I mainly wanted to get inference working and submit /something/ to see how it does. I'll elaborate more on my infrastructure/strategy/challenges in a future update.

NOTE: Version 1 was resubmitted a little later with a small fix related to cStringIO and unicode (see my post @96 on Piazza) that would cause the agent to crash if run with -Q.

*** Version 2 (submitted 4/24/2014):

Pretty much the same as V1 but with some minor bugfixes and error handling to prevent crashes during the contest. I just wanted to submit something before midnight.

*** Version 3 (submitted 4/25/2014):

Now have one defense and one offense agent. The offense agent is basically the same as before (though, unlike V2, it does not split the target food north/south (since there's only one offense agent) and just goes after the nearest food while trying very weakly to avoid ghosts. The defense agent tries to guess the most likely entry point on the border and waits there. As soon as there is an invader, it goes after it (except for a bug where doing so requires it to move to enemy territory, which it's unwilling to do) and then returns to its post.

I'm continually finding myself frustrated by having a lot of ideas for this contest and not a lot of time to implement them, but it's due soon, so hopefully I have some time this weekend. If I really fail to get to a lot of my ideas, I'll probably list them in the final README.

*** Version 4 (submitted 4/26/2014):

Offense much better now. Does pathfinding to all food using A* with maze distance as heuristic, evaluates risk of paths, chooses safest.

*** Version 5 (submitted 4/27/2014):

A very stupid bug was discovered that I think was causing poor contest performance. Here's hoping the fix to that improves things...

Beyond that, I now have improved logic for seeking capsules while being chased and other close-combat behavior.