# RLlib Multi Agent Example

An example to demonstrate the speed of RlLib.

## Single agent example

This (at `./src/singleAgent.py`) shows the performance of RlLib with tuned
hyperparameters, solving Cartpole. Key stats are as follows (for hitting a
reward of 190:

 - Iterations: 10 (4000 steps per iteration, so c. 50 episodes by the end)
 - Total time: 17 seconds (6-core CPU with GPU disabled)
