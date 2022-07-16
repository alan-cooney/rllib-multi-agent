import logging
from os.path import dirname, join, realpath

import supersuit as ss
from pettingzoo.butterfly import pistonball_v6
from ray import init, tune
from ray.rllib.agents.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

init(configure_logging=True, logging_level=logging.INFO)


class CNNModelV2(TorchModelV2, nn.Module):
    """CNN Model from RlLib example

    https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/rllib_pistonball.py_
    """

    def __init__(self, obs_space_config, act_space_config, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space_config, act_space_config,
                              num_outputs, *args, **kwargs)

        nn.Module.__init__(self)

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )

        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator():
    """Creates the Pistonball environment

    https://www.pettingzoo.ml/butterfly/pistonball

    Taken from RlLib example at 
    https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/rllib_pistonball.py
    https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b
    """
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env


env_name = "pistonball_v6"

register_env(env_name, lambda _config: ParallelPettingZooEnv(
    env_creator()))

test_env = ParallelPettingZooEnv(env_creator())
obs_space = test_env.observation_space
act_space = test_env.action_space

ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)


def gen_policy(i):
    """Generate the policy"""
    model_config = {
        "model": {
            "custom_model": "CNNModelV2",
        },
        "gamma": 0.99,
    }

    return (None, obs_space, act_space, model_config)


policies = {"policy_0": gen_policy(0)}

policy_ids = list(policies.keys())

config = PPOConfig().framework(
    framework="torch"
).environment(
    env=env_name,
    clip_actions=True
).multi_agent(
    policies=policies,
    policy_mapping_fn=(lambda agent_id: policy_ids[0])
).rollouts(
    num_rollout_workers=4,
    batch_mode="truncate_episodes",
    rollout_fragment_length=512,
    num_envs_per_worker=1,
    compress_observations=False
).training(
    lambda_=0.9,
    use_gae=True,
    gamma=0.99,
    clip_param=0.4,
    entropy_coeff=0.1,
    vf_loss_coeff=0.25,
    sgd_minibatch_size=64,
    num_sgd_iter=10,
    lr=2e-05,
    train_batch_size=512,
    grad_clip=None
).debugging(
    log_level="ERROR"
).resources(
    num_gpus=1
)

tune.run(
    "PPO",
    name="PPO",
    stop={"timesteps_total": 5000000},
    checkpoint_at_end=True,
    checkpoint_freq=10,
    local_dir=join(dirname(realpath(__file__)), ".results", "pistonball"),
    resume=False,
    config=config.to_dict()
)
