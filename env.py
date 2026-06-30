import os
import sys
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Dict

if "WEBOTS_CONTROLLER_URL" in os.environ:
    del os.environ["WEBOTS_CONTROLLER_URL"]
os.environ["WEBOTS_ROBOT_NAME"] = "swarm_supervisor"

# Ensure Python can find the Webots API on Windows
WEBOTS_PATH = "C:/Program Files/Webots"
os.environ['WEBOTS_HOME'] = WEBOTS_PATH
sys.path.append(os.path.join(WEBOTS_PATH, 'lib', 'controller', 'python'))
from controller import Supervisor

INITIAL_HEIGHT = 0.12

class WebotsDroneCoverageEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "wildfire_drone_coverage"}

    def __init__(self, num_drones=3, map_size=50, spacing=5):
        super().__init__()
        self.num_drones = num_drones
        self.map_size = map_size
        self.agents = [f"drone_{i}" for i in range(num_drones)]
        self.possible_agents = self.agents[:]
        
        #os.environ["WEBOTS_CONTROLLER_URL"] = "swarm_supervisor"
        # Connect to Webots background process
        print("making supervisor")
        self.supervisor = Supervisor()
        print("done making supervisor")
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Track individual physical drone nodes inside Webots
        print("making drones")
        self.drones = []
        for i in range(num_drones):
            if i == 0:
                drone = self.supervisor.getFromDef('drone_0')
                self.drones.append(drone)
                continue
            
            # Clean linear spacing along the X axis on the ground
            x = i * spacing
            y = 0.0
            z = INITIAL_HEIGHT
            
            drone_string = f'DEF drone_{i} Mavic2Pro {{ name "drone_{i}" translation {x} {y} {z} controller "drone_controller" }}'
            
            root = self.supervisor.getRoot()
            children_field = root.getField('children')
            children_field.importMFNodeFromString(-1, drone_string)
            
            drone = self.supervisor.getFromDef(f'drone_{i}')
            self.drones.append(drone)
        print("done making drones")

        self.initial_states = {}
        for i in range(num_drones):
            node = self.drones[i]
            if node is not None:
                self.initial_states[i] = {
                    "translation": node.getField("translation").getSFVec3f(),
                    "rotation": node.getField("rotation").getSFRotation()
                }

        # Coverage Grid: 50x50 matrix (0 = Unvisited, 1 = Covered)
        self.coverage_grid = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        # MARL Spaces: Local observations per drone
        # [Altitude, Velocity_X, Velocity_Y, Radar_Obstacle_Distance]
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Action Spaces: Target flight velocities [vx, vy, vz]
        self.action_spaces = {
            agent: Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)
            for agent in self.agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        for i in range(self.num_drones):
            node = self.drones[i]
            if node is not None:
                node.resetPhysics()
                node.getField("translation").setSFVec3f(self.initial_states[i]["translation"])
                node.getField("rotation").setSFRotation(self.initial_states[i]["rotation"])

        self.supervisor.simulationResetPhysics()

        self.coverage_grid.fill(0)

        self.supervisor.step(self.timestep)
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_observations(self):
        obs = {}
        for i, drone in enumerate(self.drones):
            pos = drone.getPosition()   # [X, Y, Z]
            vel = drone.getVelocity()   # [Vx, Vy, Vz, Wx, Wy, Wz]
        
            obs[i] = np.array([pos[2], vel[0], vel[1], 1.0], dtype=np.float32)
        return obs

    def step(self, actions):
        for i, action in enumerate(actions):
            node = self.drones[i]
            # Convert the action array [vx, vy, vz] to a Webots 3D vector command
            # Note: This updates the velocity field directly
            node.setVelocity([float(action[0]), float(action[1]), float(action[2]), 0, 0, 0])

        for _ in range(4): 
            self.supervisor.step(self.timestep)

        newly_covered_cells = 0
        for i, drone in enumerate(self.drones):
            pos = drone.getPosition()
            grid_x = int(np.clip((pos[0] + 25), 0, self.map_size - 1))
            grid_y = int(np.clip((pos[1] + 25), 0, self.map_size - 1))
            
            if self.coverage_grid[grid_x, grid_y] == 0:
                self.coverage_grid[grid_x, grid_y] = 1
                newly_covered_cells += 1

        team_reward = float(newly_covered_cells * 10.0)
        rewards = {agent: team_reward for agent in self.agents}

        crashed = any(d.getPosition()[2] < INITIAL_HEIGHT + 0.1 for d in self.drones)
        terminations = {agent: crashed for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        observations = self._get_observations()
        return observations, rewards, terminations, truncations, {agent: {} for agent in self.agents}