import ray
from ray import tune
from ray.rllib.agents.ppo import PPOConfig

# Initialize ray
ray.init(configure_logging=False,
         logging_level="info")

# Model overrides
# From https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/cartpole-ppo.yaml
model = {
    "fcnet_hiddens": [32],
    "fcnet_activation": "linear",
    "vf_share_layers": True
}

# Config overrides
# From https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/cartpole-ppo.yaml
config = PPOConfig().framework(
    framework="torch"
).training(
    gamma=0.99,
    lr=0.0003,
    vf_loss_coeff=0.01,
    num_sgd_iter=6,
    model=model,
    train_batch_size=4000
).rollouts(
    observation_filter="MeanStdFilter",
    num_rollout_workers=5
).environment(
    env="CartPole-v0"
).resources(
    # num_gpus=1
)

# Setup the trainer
# trainer = PPOTrainer(env="CartPole-v0", config=config.to_dict())

# analysis = tune.run(
#     "PPO",
#     stop={"episode_reward_mean": 190},
#     config=config.to_dict(),
#     checkpoint_at_end=True)

analysis = tune.run(
    "PPO",
    stop={"episode_reward_mean": 190},
    config=config.to_dict(),
    checkpoint_at_end=True
)

# or simply get the last checkpoint (with highest "training_iteration")
# last_checkpoint = analysis.get_last_checkpoint()
