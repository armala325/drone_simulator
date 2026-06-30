from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from env import WebotsDroneCoverageEnv

def main():
    print("s1")
    raw_env = WebotsDroneCoverageEnv(num_drones=3)
    print("e1")

    print("s2")
    env_wrapper = PettingZooWrapper(
        env=raw_env,
        categorical_actions=False
    )
    print("e2")

    print("s3")
    mappo_config = MappoConfig(
        share_param_critic=True,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        critic_coef=0.5
    )
    print("e3")

    print("s4")
    actor_mlp = MlpConfig(num_cells=[128, 128], activation_class="torch.nn.ReLU")
    critic_mlp = MlpConfig(num_cells=[256, 256], activation_class="torch.nn.ReLU")
    print("e4")

    print("s5")
    experiment_config = ExperimentConfig(
        sampling_device="cpu",     # Keeps Webots CPU interactions clean
        train_device="cuda",       # Offloads heavy multi-agent PPO gradients to your GPU
        max_stages=50,             # Total training iterations
        samplers_per_worker=1,
        epoch_per_stage=4          # PPO update epochs per step collection
    )
    print("e5")

    print("s6")
    experiment = Experiment(
        env=env_wrapper,
        algorithm_config=mappo_config,
        model_config=actor_mlp,
        critic_model_config=critic_mlp,
        seed=42,
        config=experiment_config
    )
    print("e6")

    print("Starting Multi-Agent Drone Training in Webots via BenchMARL...")
    experiment.run()

if __name__ == "__main__":
    main()