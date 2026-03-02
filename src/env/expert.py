from __future__ import annotations

from typing import List, Optional, Tuple

from minigrid.core.actions import Actions
from minigrid.core.world_object import Goal

DIR_TO_VEC = {
    0: (1, 0),   # right
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}


def _find_goal_pos(env_unwrapped) -> Tuple[int, int]:
    if hasattr(env_unwrapped, "goal_pos") and env_unwrapped.goal_pos is not None:
        return tuple(env_unwrapped.goal_pos)

    grid = env_unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if isinstance(obj, Goal):
                return (x, y)
    raise RuntimeError("Goal position not found in grid")


def _turn_actions(current_dir: int, target_dir: int) -> List[int]:
    if current_dir == target_dir:
        return []

    right_steps = (target_dir - current_dir) % 4
    left_steps = (current_dir - target_dir) % 4

    if right_steps <= left_steps:
        return [int(Actions.right)] * right_steps
    return [int(Actions.left)] * left_steps


def plan_oracle_actions(env_unwrapped) -> List[int]:
    """
    Deterministic shortest path in EmptyEnv (no obstacles):
    rotate to x-axis target, move forward, then y-axis target.
    """
    agent_x, agent_y = tuple(env_unwrapped.agent_pos)
    agent_dir = int(env_unwrapped.agent_dir)
    goal_x, goal_y = _find_goal_pos(env_unwrapped)

    actions: List[int] = []

    while (agent_x, agent_y) != (goal_x, goal_y):
        if agent_x != goal_x:
            target_dir = 0 if goal_x > agent_x else 2
        else:
            target_dir = 1 if goal_y > agent_y else 3

        turns = _turn_actions(agent_dir, target_dir)
        actions.extend(turns)
        agent_dir = target_dir

        actions.append(int(Actions.forward))
        dx, dy = DIR_TO_VEC[agent_dir]
        agent_x += dx
        agent_y += dy

    return actions


def oracle_action(env_unwrapped) -> int:
    plan = plan_oracle_actions(env_unwrapped)
    if not plan:
        return int(Actions.done)
    return int(plan[0])


def evaluate_expert_success(env_id: str, episodes: int, seed: int, tile_size: int = 8, max_steps: Optional[int] = None) -> float:
    from src.env.make_env import make_env

    success = 0
    for ep in range(episodes):
        env, obs, _ = make_env(env_id=env_id, seed=seed + ep, tile_size=tile_size, max_steps=max_steps)
        done = False
        while not done:
            action = oracle_action(env.unwrapped)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward > 0:
                success += 1
        env.close()
    return success / max(1, episodes)
