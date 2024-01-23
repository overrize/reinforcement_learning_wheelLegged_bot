from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg,LeggedRobotCfgPPO

class ELWRoughCfg( LeggedRobotCfg ):
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

    class control( LeggedRobotCfg.control ):
        # PD Drive paramters:
        control_type = 'P'
        # 刚度
        stiffness = {'joint': 20.}  # [N*m/rad]
        # 阻尼
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * sction + defaultAngle
        action_scale = 0.25
        #decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_leg_rb/urdf/wheel_leg_rb.urdf'
        name = "wheel_leg_rb"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class reward( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class ELWRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_elw'
        
