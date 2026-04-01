# 📘 Algebraic multigrid GBP playground

## 🛠️ Installation

To get started, install the necessary dependencies with the following command:

```bash
# Example for Python projects
pip install -r requirements.txt
```

Make sure you are using the appropriate environment (Python 3.10)

---

## 🖥️ Using the GUI

Once installed, you can launch the GUI for the 2D posegraph example:

```bash
python 2d_posegraph.py
```

Or you can launch iterative 2D SLAM example (with simple 2D constraints)

```bash
python 2d_slam.py
```

## 2D Posegraph usage:

Variable nodes are pixel locations with factors between variables being a 2D measured distance. Certain *anchor nodes* will have a strong prior on its ground truth location to anchor the problem.

* **LEFT MOUSE BUTTON**: Will generate a variable node at the location clicked.
* **RIGHT MOUSE BUTTON**: Pan the graph view.
* **SCROLL WHEEL**: Zoom in/out.
---
* **RESET**: Resets the factor graph to its initial state (if you've already run an optimisation then it will show the error stats for that run)
* **GENERATE RANDOM**: Generates N number of random nodes in the workspace (amount is the number to the left). These variables will initialise to the centre of the screen and will have nearest neighbour connections to its ground truth neighbours (there is a way to make default starting position a noisy value of the ground truth but I can't remember where I set that)
* **CLEAR NODES**: Removes all variables nodes from the factor graph.
* **MEANING OF LIFE**: Special custom variable node configuration (can use instead of 'generate random' for a repeatable test)
* **TOGGLE MULTIGRID**: Toggles whether to use multigrid or not (default will be my implementation of AMG). Think you need to set this before running the optimisation.
* **USE PyAMG**: Will use pyamg to generate the levels.
* **TOGGLE WILDFIRE**: Will use 'wildfire' message passing instead of a synchronous schedule 
* **RUN GRAPH**: Will start the optimisation running
* **SAVE TO FILE**: Not sure what its saving, will have to test
---
* **LAYERS**: If using multigrid you can cycle through the layers in the visualisation.
* **PLAY/PAUSE/SKIP**: Start, pause and step forward one iteration. You may have to click a lot of time for it to actually pause, just keep spamming the button, FPS will go to 30 once its actually paused.


## 2D SLAM usage:

A simple robot with a visible radius moves around its environment (creating odometry factors) and records distances to any landmarks it sees (landmark factors) and optimises for its trajectory (pose variables).

* **MOVEMENT**: Use W,A,S,D to move the robot around the workspace. Q lowers speed, E increases speed.
* **LEFT MOUSE BUTTON**: Will generate a landmark at the location clicked.
* **RIGHT MOUSE BUTTON**: Pan the graph view.
* **SCROLL WHEEL**: Zoom in/out.
---
* **RESET**: Resets the factor graph to its initial state
* **GENERATE RANDOM**: Generates N number of random landmarks in the workspace. Clicking this again will add another random set densifying the problem.
* **CLEAR NODES**: Removes all landmarks from the workspace.
* **MEANING OF LIFE**: Doesn't do anything (don't press)
* **TOGGLE MULTIGRID**: Toggles whether to use multigrid or not (default will be my implementation of AMG) Think you need to set this before running the optimisation.
* **USE PyAMG**: Will use pyamg to generate the levels.
* **TOGGLE WILDFIRE**: Will use 'wildfire' message passing instead of a synchronous schedule 
* **RUN GRAPH**: Will start the optimisation running
* **SAVE TO FILE**: Not sure what its saving, will have to test
---
* **LAYERS**: If using multigrid you can cycle through the layers in the visualisation.
* **ACTIVE**: Can't remember what this does!
* **PLAY/PAUSE/SKIP**: Start, pause and step forward one iteration

## 📌 Additional Instructions

Good luck! Lots of things are probably broken so apologies.

---

## 🙋 Support

If you have questions or encounter issues, send me a message - @callum-rhodes

