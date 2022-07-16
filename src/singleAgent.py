from os.path import dirname, join, realpath

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOConfig

# Initialize ray
ray.init(configure_logging=False,
         logging_level="info")

# Model overrides
# https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/cartpole-ppo.yaml
model = {
    "fcnet_activation": "linear",
    "fcnet_hiddens": [32],
    "vf_share_layers": True,
}

# Config overrides
# https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/cartpole-ppo.yaml
config = PPOConfig().framework(
    framework="torch"
).training(
    gamma=0.99,
    lr=0.0003,
    model=model,
    num_sgd_iter=6,
    train_batch_size=4000,
    vf_loss_coeff=0.01,
).rollouts(
    num_rollout_workers=5,
    observation_filter="MeanStdFilter",
).environment(
    env="CartPole-v0"
).resources(
    # num_gpus=1
)

# Train (note we use Ray Tune to enable hyperparmeter tuning in other
# experiments, but here we only have one parameter).
tune.run(
    "PPO",
    checkpoint_at_end=True,
    config=config.to_dict(),
    local_dir=join(dirname(realpath(__file__)), ".results"),
    resume=False,
    stop={"episode_reward_mean": 190},
)
