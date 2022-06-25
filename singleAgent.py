import ray
from ray import tune

ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_workers": 5,
        # "num_gpus": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "framework": "tf2"
    },
)
