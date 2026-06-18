"""Environment factory — single source of truth for env creation.

Usage:
    from utils.env_factory import make_env, ENV_REGISTRY

    env, env_info = make_env("ant")
    # env_info.obs_dim, env_info.goal_start_idx, env_info.goal_end_idx, env_info.action_size
"""

from dataclasses import dataclass
from typing import Tuple

from brax import envs


@dataclass(frozen=True)
class EnvInfo:
    obs_dim: int
    goal_start_idx: int
    goal_end_idx: int


ENV_REGISTRY = {
    "reacher":          {"module": "envs.reacher",                    "cls": "Reacher",          "kwargs": {"backend": "spring"}, "obs_dim": 10,  "goal": (4, 7)},
    "pusher":           {"module": "envs.pusher",                     "cls": "Pusher",           "kwargs": {"backend": "spring"}, "obs_dim": 20,  "goal": (10, 13)},
    "ant":              {"module": "envs.ant",                        "cls": "Ant",              "kwargs": {"backend": "spring", "exclude_current_positions_from_observation": False, "terminate_when_unhealthy": True}, "obs_dim": 29, "goal": (0, 2)},
    "ant_ball":         {"module": "envs.ant_ball",                   "cls": "AntBall",          "kwargs": {"backend": "spring", "exclude_current_positions_from_observation": False, "terminate_when_unhealthy": True}, "obs_dim": 31, "goal": (28, 30)},
    "ant_push":         {"module": "envs.ant_push",                   "cls": "AntPush",          "kwargs": {"backend": "mjx"}, "obs_dim": 31, "goal": (0, 2)},
    "humanoid":         {"module": "envs.humanoid",                   "cls": "Humanoid",         "kwargs": {"backend": "spring", "exclude_current_positions_from_observation": False, "terminate_when_unhealthy": True}, "obs_dim": 268, "goal": (0, 3)},
    "arm_reach":        {"module": "envs.manipulation.arm_reach",     "cls": "ArmReach",         "kwargs": {"backend": "mjx"}, "obs_dim": 13, "goal": (7, 10)},
    "arm_grasp":        {"module": "envs.manipulation.arm_grasp",     "cls": "ArmGrasp",         "kwargs": {"backend": "mjx", "cube_noise_scale": 0.3}, "obs_dim": 23, "goal": (16, 23)},
    "arm_push_easy":    {"module": "envs.manipulation.arm_push_easy", "cls": "ArmPushEasy",      "kwargs": {"backend": "mjx"}, "obs_dim": 17, "goal": (0, 3)},
    "arm_push_hard":    {"module": "envs.manipulation.arm_push_hard", "cls": "ArmPushHard",      "kwargs": {"backend": "mjx"}, "obs_dim": 17, "goal": (0, 3)},
    "arm_binpick_easy": {"module": "envs.manipulation.arm_binpick_easy", "cls": "ArmBinpickEasy", "kwargs": {"backend": "mjx"}, "obs_dim": 17, "goal": (0, 3)},
    "arm_binpick_hard": {"module": "envs.manipulation.arm_binpick_hard", "cls": "ArmBinpickHard", "kwargs": {"backend": "mjx"}, "obs_dim": 17, "goal": (0, 3)},
    "arm_binpick_easy_EEF": {"module": "envs.manipulation.arm_binpick_easy_EEF", "cls": "ArmBinpickEasyEEF", "kwargs": {"backend": "mjx"}, "obs_dim": 11, "goal": (0, 3)},
}

# Maze environments use pattern matching
_MAZE_PATTERNS = [
    # (prefix, module, cls, obs_dim, goal_start, goal_end)
    ("ant_",      "envs.ant_maze",    "AntMaze",      29, 0, 2),
    ("humanoid_", "envs.humanoid_maze", "HumanoidMaze", 268, 0, 3),
]


def make_env(env_id: str):
    """Create an environment by id.

    Returns (env, EnvInfo). Raises KeyError if env_id not found.
    """
    import importlib

    # Direct lookup
    if env_id in ENV_REGISTRY:
        entry = ENV_REGISTRY[env_id]
        mod = importlib.import_module(entry["module"])
        cls = getattr(mod, entry["cls"])
        env = cls(**entry["kwargs"])
        info = EnvInfo(obs_dim=entry["obs_dim"], goal_start_idx=entry["goal"][0], goal_end_idx=entry["goal"][1])
        return env, info

    # Maze pattern matching
    for prefix, module, cls_name, obs_dim, gs, ge in _MAZE_PATTERNS:
        if env_id.startswith(prefix) and "maze" in env_id:
            layout = env_id[len(prefix):]
            # Generalization envs
            if "gen" in layout:
                if "ant" not in prefix:
                    raise NotImplementedError(f"Generalization only for ant mazes: {env_id}")
                from envs.ant_maze_generalization import AntMazeGeneralization
                gen_idx = layout.find("gen")
                maze_layout_name = layout[:gen_idx - 1]
                generalization_config = layout[gen_idx + 4:]
                env = AntMazeGeneralization(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=maze_layout_name,
                    generalization_config=generalization_config)
                info = EnvInfo(obs_dim=obs_dim, goal_start_idx=gs, goal_end_idx=ge)
                return env, info

            mod = importlib.import_module(module)
            cls = getattr(mod, cls_name)
            env = cls(backend="spring", exclude_current_positions_from_observation=False,
                      terminate_when_unhealthy=True, maze_layout_name=layout) if "ant" in prefix else \
                  cls(backend="spring", maze_layout_name=layout)
            info = EnvInfo(obs_dim=obs_dim, goal_start_idx=gs, goal_end_idx=ge)
            return env, info

    # arm_grasp with noise scale
    if env_id.startswith("arm_grasp") and len(env_id) > 9:
        scale = float(env_id[10:])
        mod = importlib.import_module("envs.manipulation.arm_grasp")
        env = mod.ArmGrasp(cube_noise_scale=scale, backend="mjx")
        info = EnvInfo(obs_dim=23, goal_start_idx=16, goal_end_idx=23)
        return env, info

    available = sorted(ENV_REGISTRY.keys())
    raise KeyError(f"Unknown env_id: '{env_id}'. Available: {available}")


def wrap_env(env, episode_length: int = 1000):
    """Wrap environment for training."""
    return envs.training.wrap(env, episode_length=episode_length)
