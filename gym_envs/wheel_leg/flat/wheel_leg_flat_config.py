import torch
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


#机体配置
class wheel_leg_flatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 10
        num_observations = 229 #观测员数量

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0,0.0,0.50] # x,y,z [m] 坐标
        default_joint_angles = { # 单位 [rad]
            'left_hipL_joint':0.1,
            'left_kneelL_joint':-0.1,
            'left_wheel_joint':0.1,
            'left_hipR_joint':-0.1,
            'left_kneelR_joint':0.1,

            'right_hipL_joint':0.1,
            'right_kneelL_joint':-0.1,
            'right_hipR_joint':-0.1,
            'right_kneelR_joint':0.1,
            'right_wheel_joint':0.1,
        }

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class control( LeggedRobotCfg.control ):
        # PD Drive paramters:
        control_type = 'T'
        # 刚度
        stiffness = {
            'left_hipL_joint':30.,
            'left_kneelL_joint':0.,
            'left_wheel_joint':30.,
            'left_hipR_joint':30.,
            'left_kneelR_joint':0.,

            'right_hipL_joint':30.,
            'right_kneelL_joint':0.,
            'right_hipR_joint':30.,
            'right_kneelR_joint':0.,
            'right_wheel_joint':30.,
            }  # [N*m/rad]
        # 阻尼
        damping = {
            'left_hipL_joint':5.,
            'left_kneelL_joint':0.,
            'left_wheel_joint':5.,
            'left_hipR_joint':5.,
            'left_kneelR_joint':0.,

            'right_hipL_joint':5.,
            'right_kneelL_joint':0.,
            'right_hipR_joint':5.,
            'right_kneelR_joint':0.,
            'right_wheel_joint':5.,
            }     # [N*m*s/rad]
        # action scale: target angle = actionScale * sction + defaultAngle
        action_scale = 1.0
        #decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_leg_rb/urdf/wheel_leg_rb.xml'
        name = "wheel_leg_rb"
        foot_name = "wheel"
        penalize_contacts_on = ["left_hipL_link", "left_kneelL_link" , "left_hipR_link","left_kneelR_link","right_hipL_link","right_kneelL_link","right_hipR_link","right_kneelR_link"]
        terminate_after_contacts_on = ["base_link"]
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class reward( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            
class wheel_leg_flatPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01  #熵
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'wheel_leg_2024'
        experiment_name = 'flat_wheel_leg'
        max_iterations = 30
    



    

