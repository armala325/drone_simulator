from typing import Callable, Dict, List, Optional
import torch
#from tensordict import TensorDictBase
from torchrl.data import Composite
#from torchrl.envs import EnvBase, PettingZooWrapper
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig

# Import TaskClass instead of Task
from benchmarl.environments.common import TaskClass

#import numpy as np
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, UnboundedContinuous, Bounded, UnboundedDiscrete
from benchmarl.environments.common import TaskClass

# =====================================================================
# 1. CREATE A BENCHMARL-COMPATIBLE CUSTOM WRAPPER FOR YOUR WEBOTS ENV
# =====================================================================
class BenchMarlWebotsWrapper(EnvBase):
    def __init__(self, pz_env, device="cpu"):
        # BenchMARL uses an empty root batch size for the environment level
        super().__init__(device=device, batch_size=torch.Size([]))
        self.pz_env = pz_env
        self.num_drones = pz_env.num_drones
        self.agents = pz_env.agents
        
        # Sample an observation and action to infer shape sizes automatically
        sample_obs = pz_env.observation_space(self.agents[0]).sample()
        sample_act = pz_env.action_space(self.agents[0]).sample()
        self.obs_dim = sample_obs.shape[0]
        self.action_dim = sample_act.shape[0]
        
        # 1. Observation Spec: Structured under the "agents" group name
        self.observation_spec = Composite({
            "agents": Composite({
                "observation": UnboundedContinuous(
                    shape=torch.Size([self.num_drones, self.obs_dim]),
                    device=device
                )
            }, batch_size=torch.Size([self.num_drones]))
        })
        
        # 2. Action Spec: Structured under the "agents" group name with bounds
        act_space = pz_env.action_space(self.agents[0])
        low = torch.tensor(act_space.low, device=device).unsqueeze(0).expand(self.num_drones, -1)
        high = torch.tensor(act_space.high, device=device).unsqueeze(0).expand(self.num_drones, -1)
        
        self.action_spec = Composite({
            "agents": Composite({
                "action": Bounded(
                    low=low, high=high,
                    shape=torch.Size([self.num_drones, self.action_dim]),
                    device=device
                )
            }, batch_size=torch.Size([self.num_drones]))
        })
        
        # 3. Reward Spec: Stacked reward per agent group
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": UnboundedContinuous(
                    shape=torch.Size([self.num_drones, 1]),
                    device=device
                )
            }, batch_size=torch.Size([self.num_drones]))
        })
        
        # 4. Done/Terminal Specs (Global tracking at the root level)
        self.done_spec = Composite({
            "done": UnboundedDiscrete(shape=torch.Size([1]), dtype=torch.bool, device=device),
            "terminated": UnboundedDiscrete(shape=torch.Size([1]), dtype=torch.bool, device=device)
        })

    def _reset(self, tensordict=None):
        # Reset the underlying custom Webots PettingZoo environment
        obs_dict = self.pz_env.reset()[0]
        
        # Stack individual drone observations into a single multi-agent tensor [num_drones, obs_dim]
        obs_list = [torch.tensor(obs_dict[agent], dtype=torch.float32, device=self.device) for agent in self.agents]
        stacked_obs = torch.stack(obs_list, dim=0)
        
        return TensorDict({
            "agents": TensorDict({"observation": stacked_obs}, batch_size=torch.Size([self.num_drones]))
        }, batch_size=torch.Size([]))

    def _step(self, tensordict):
        # 1. Pull the stacked action tensor from MAPPO [num_drones, action_dim]
        action_tensor = tensordict["agents", "action"]
        
        # 2. Convert to an ordered list of numpy actions for your custom env step method
        actions_list = [action_tensor[i].cpu().numpy() for i in range(self.num_drones)]
        
        # 3. Step the environment
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.pz_env.step(actions_list)
        
        # 4. Collect and restack observations and rewards into BenchMARL format
        obs_list = [torch.tensor(obs_dict[agent], dtype=torch.float32, device=self.device) for agent in self.agents]
        reward_list = [torch.tensor([reward_dict[agent]], dtype=torch.float32, device=self.device) for agent in self.agents]
        
        stacked_obs = torch.stack(obs_list, dim=0)
        stacked_reward = torch.stack(reward_list, dim=0)
        
        # Determine episode completion conditions
        done = any(terminated_dict.values()) or any(truncated_dict.values())
        terminated = any(terminated_dict.values())
        
        return TensorDict({
            "agents": TensorDict({
                "observation": stacked_obs,
                "reward": stacked_reward
            }, batch_size=torch.Size([self.num_drones])),
            "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device)
        }, batch_size=torch.Size([]))

    def _set_seed(self, seed: Optional[int]):
        pass


# =====================================================================
# 2. UPDATE YOUR TASK CLASS TO USE THE NEW WRAPPER
# =====================================================================
class WebotsDroneCoverageTask(TaskClass):
    def __init__(self, name: str = "wildfire_drone_coverage", config: dict = None):
        if config is None:
            config = {"num_drones": 3, "map_size": 50, "spacing": 5}
        super().__init__(name=name, config=config)

    @staticmethod
    def env_name() -> str:
        return "webots_drone_env"

    def get_env_fun(
        self, num_envs: int, continuous_actions: bool, seed: Optional[int], device: str
    ) -> Callable[[], EnvBase]:
        from env import WebotsDroneCoverageEnv
        
        def make_env():
            # Instantiate raw environment
            raw_env = WebotsDroneCoverageEnv(
                num_drones=self.config.get("num_drones", 3),
                map_size=self.config.get("map_size", 50),
                spacing=self.config.get("spacing", 5),
            )
            # Use our custom wrapper instead of PettingZooWrapper
            return BenchMarlWebotsWrapper(raw_env, device=device)
            
        return make_env

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return 1000

    def has_render(self, env: EnvBase) -> bool:
        return False

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"agents": [f"drone_{i}" for i in range(self.config.get("num_drones", 3))]}

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        return env.action_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

# 2. Instantiate your custom task object
task = WebotsDroneCoverageTask()

# 3. Set up your configurations using the .get_from_yaml() defaults
algorithm_config = MappoConfig.get_from_yaml()
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()
experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.loggers = ["csv"]
experiment_config.clip_grad_norm = True

# 4. Initialize the Experiment correctly using 'task' instead of 'env'
experiment = Experiment(
    task=task,  # <-- Changed from env=env to task=task
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    config=experiment_config,
    seed=42
)

# 5. Run your training loop
experiment.run()