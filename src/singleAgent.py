from os.path import dirname, join, realpath

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOConfig, PPOTrainer

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

# Environment to use
ENV_NAME = "CartPole-v0"

# Experiment name (used for checkpoint & video directories)
EXPERIMENT_NAME = "single-player-cartpole"

# Horizon (number of steps to run for)
HORIZON = 1000

# Config overrides
# https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/cartpole-ppo.yaml
config = PPOConfig().training(
    gamma=0.99,
    lr=0.0003,
    model=model,
    num_sgd_iter=6,
    train_batch_size=4000,
    vf_loss_coeff=0.01,
).rollouts(
    num_rollout_workers=5,
    observation_filter="MeanStdFilter",
    horizon=HORIZON
).environment(
    env=ENV_NAME
)


def train() -> str:
    """Train the model

    Returns:
        str: Directory for the last checkpoint
    """
    # Nnote we use Ray Tune to enable hyperparmeter tuning in other
    # experiments, but here we only have one parameter.
    analysis = tune.run(
        "PPO",
        name=EXPERIMENT_NAME,
        checkpoint_at_end=True,
        config=config.to_dict(),
        local_dir=join(dirname(realpath(__file__)), "../.results"),
        resume="AUTO",
        stop={"episode_reward_mean": HORIZON - 10}
    )

    # Return the directory of the last checkpoint
    return analysis.get_last_checkpoint()


def record_video(last_checkpoint_dir: str) -> None:
    """Record a video of the model running

    Args:
        last_checkpoint_dir (string): _Directory for the best checkpoint
    """
    # Set the video dir
    video_dir = join(dirname(realpath(__file__)),
                     "../.videos", EXPERIMENT_NAME)

    # Update the config to set record_env to true
    record_config = config.environment(
        record_env=video_dir, env=ENV_NAME).to_dict()

    # Run one iteration so we record some videos
    trainer = PPOTrainer(env=ENV_NAME, config=record_config)
    trainer.restore(last_checkpoint_dir)
    trainer.train()


if __name__ == "__main__":
    # Train the model
    checkpoint_dir = train()

    # Record a video of the model in action
    record_video(checkpoint_dir)
