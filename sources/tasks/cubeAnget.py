from isaacgym import gymtorch
from isaacgym import gymapi
import torch
from isaacgymenvs.utils.torch_jit_utils import tensor_clamp 
from .base.vec_task import VecTask
import numpy as np
import os

class CubeAgent(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
      self.cfg = cfg
      self.max_episode_length = 500
      self.cfg["env"]["numObservations"] = 6
      self.cfg["env"]["numActions"] = 3

      super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
      self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
      vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 13)
      self.root_states = vec_root_tensor
      self.root_positions = vec_root_tensor[..., 0:3]
      self.root_linvels = vec_root_tensor[..., 7:10]
      self.root_quats = vec_root_tensor[..., 4:7]
      self.root_angvels = vec_root_tensor[..., 11:12]
      self.gym.refresh_actor_root_state_tensor(self.sim)
      self.initial_root_states = vec_root_tensor.clone()

      max_thrust = 2
      bodies_per_env = 3
      self.thrust_lower_limits = -max_thrust * torch.ones(3, device=self.device, dtype=torch.float32)
      self.thrust_upper_limits = max_thrust * torch.ones(3, device=self.device, dtype=torch.float32)
      self.thrusts = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
      self.forces = torch.zeros((self.num_envs, bodies_per_env), dtype=torch.float32, device=self.device, requires_grad=False)

      # 调试过程中的观测窗口的自身位置和朝向的目标点位置
      if self.viewer:
        cam_pos = gymapi.Vec3(1.0, 1.0, 1.8)
        cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        #建立地面
        self._create_ground_plane() 
        #建立环境并行运行
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, 0.0)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../assets/urdf')
        asset_file = "cube.urdf"
        asset_options = gymapi.AssetOptions()
        default_pose = gymapi.Transform()
        cube_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.cube_handles = []
        self.envs = []
        for i in range(self.num_envs):
            #create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
            cube_handle = self.gym.create_actor(env, cube_asset, default_pose, "cube", i, 1, 0)
            self.cube_handles.append(cube_handle)
    
    def pre_physics_step(self, actions):
        #reset enviroments
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        actions = actions.to(self.device)
        thrust_action_speed_sacle = 20
        self.thrusts += self.dt * thrust_action_speed_sacle * actions[:,0:3]
        self.thrusts[:] = tensor_clamp(self.thrusts, self.thrust_lower_limits,self.thrust_upper_limits)
        self.forces[:, 0] = self.thrusts[:, 0]
        self.forces[:, 1] = self.thrusts[:, 1]
        self.forces[:, 2] = self.thrusts[:, 2]

        #clear actions for reset envs
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)



    def compute_observation(self):
        target_x = 0.0
        target_y = 0.0
        target_z = 1.0
        self.obs_buf[..., 0] = (target_x - self.root_positions[..., 0]) / 3
        self.obs_buf[..., 1] = (target_y - self.root_positions[..., 1]) / 3
        self.obs_buf[..., 2] = (target_z - self.root_positions[..., 2]) / 3
        self.obs_buf[..., 3:6] = self.root_linvels / 2
        return self.obs_buf
    
    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_cube_reward(
            self.root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.gym.refresh_actor_root_state_tensor(self.sim) 
        self.compute_observation()
        self.compute_reward()

@torch.jit.script
def compute_cube_reward(root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    # distance to target
    target_dist = torch.sqrt(root_positions[..., 0] * root_positions[..., 0] +
                             root_positions[..., 1] * root_positions[..., 1] +
                             (1 - root_positions[..., 2]) * (1 - root_positions[..., 2]))
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)
    # combined reward
    reward = pos_reward 
    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 3.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.3, ones, die)
    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    return reward, reset
        

