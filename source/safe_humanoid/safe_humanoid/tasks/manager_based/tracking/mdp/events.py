"""Initialize the robot at the first retargeted reference pose."""


def reset_to_reference_motion(env, env_ids, command_name: str = "motion") -> None:
    command = env.command_manager.get_term(command_name)
    # Reset events run before episode_length_buf is cleared, so access frame 0
    # directly instead of the current per-environment command property.
    robot = env.scene[command.cfg.asset_name]
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] = command._root_pos[0] + env.scene.env_origins[env_ids]
    root_state[:, 3:7] = command._root_quat[0]
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        command._joint_pos[0].expand(len(env_ids), -1),
        command._joint_vel[0].expand(len(env_ids), -1),
        env_ids=env_ids,
    )
