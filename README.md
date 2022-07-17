# RLlib Multi Agent Example

An example to demonstrate the speed & simplicity of RlLib. Each example trains a
model, outputting results to `.results` and videos of the last iteration to
`.videos`.

## Setup

This repository includes a [devcontianer](https://containers.dev/) for one-click
setup. You can also install manually (`pip install .`).

## Models

### Cartpole (Single agent example)

Cartpole is included as a comparison against multi-agent examples.

```bash
python ./src/singleAgent.py
```

This shows the performance of RlLib with tuned hyperparameters, solving
Cartpole. Key stats are as follows (for hitting a reward of 990):

 - Iterations: 42 (4000 steps per iteration, so c. 50 episodes by the end)
 - Total time: 50 seconds (6-core CPU with GPU disabled)

### Pistonball (Multi-agent example)

```bash
python ./src/multiAgent.py
```
