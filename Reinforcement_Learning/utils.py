import numpy as np

from typing import Callable, List, Tuple
from agent import Agent
from world import World

from visualizations import show_policy

SEED = 123
ACTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


def get_agent_policy(world: World, agent: Agent, alpha: float,
                     epsilon: float, gamma: float, decay: float,
                     decay_rate: int, amount_of_episodes: int,
                     update_function: Callable, max_steps: int):
    value_function_q, converged, episodes, path_length = _get_value_function_q(
        world, agent, ACTIONS, alpha,
        epsilon, gamma, decay, decay_rate,
        amount_of_episodes, SEED,
        update_function, max_steps
    )

    policy = _compute_agent_policy(world, value_function_q)

    return value_function_q, policy, converged, episodes, path_length


def _get_value_function_q(world: World, agent: Agent,
                          actions: List[Tuple[int, int]],
                          alpha: float, epsilon: float,
                          gamma: float, decay: float,
                          decay_rate: int, amount_of_episodes: int,
                          seed: int, update_function: Callable,
                          max_steps: int):
    if decay < 1.0:
        decay_interval = int(amount_of_episodes / decay_rate)
    else:
        decay_interval = amount_of_episodes

    # Step 1 / 1
    value_function_q = np.zeros((*world.size, len(actions)))
    random_generator = np.random.default_rng(seed)
    converged = False
    episode = 0
    path_length = -1

    # Step 2 / 2
    for episode in range(1, amount_of_episodes + 1):
        converged, path_length = optimize_episode(
            value_function_q, episode, world, agent, actions,
            alpha, epsilon, gamma,
            decay, decay_interval,
            random_generator, update_function, max_steps
        )
        if converged:
            break

    return value_function_q, converged, episode, path_length


def optimize_episode(value_function_q: np.ndarray, episode: int,
                     world: World, agent: Agent,
                     actions: List[Tuple[int, int]], alpha: float,
                     epsilon: float, gamma: float, decay: float,
                     decay_interval: int, random_generator,
                     update_function: Callable, max_steps: int):
    # Step 3 / 3
    curr_state = agent.state

    # Step 4 / Non-existence
    curr_action_id = _choose_action(
        value_function_q, curr_state, random_generator, epsilon, actions
    )
    curr_action = actions[curr_action_id]

    # Step 5 / 4
    for step in range(max_steps):
        if agent.is_terminal(curr_state):
            break

        # Steps 6 & 7 / 5 & 6
        next_state, reward = agent.check_action(curr_state, curr_action)
        next_action_id = _choose_action(
            value_function_q, next_state, random_generator, epsilon, actions
        )
        next_action = actions[next_action_id]

        # Step 8 / 7
        value_function_q[curr_state[0], curr_state[1], curr_action_id] = update_function(
            value_function_q, reward, curr_state, alpha, gamma,
            next_state, curr_action_id, next_action_id
        )

        # Step 9 / 8
        curr_state = next_state
        curr_action = next_action
        curr_action_id = next_action_id

    epsilon_threshold = 0.001
    if episode % decay_interval == 0 and epsilon > epsilon_threshold:
        epsilon *= decay

    policy = _compute_agent_policy(world, value_function_q)

    converged, path_length = _test_policy(
        policy, world, agent, actions
    )

    return converged, path_length


def _choose_action(value_function_q: np.ndarray, curr_state: np.ndarray,
                   random_generator, epsilon: float,
                   actions: List[Tuple[int, int]]):
    # Exploration
    if random_generator.random() < epsilon:
        return random_generator.choice(len(actions))

    # Exploitation
    else:
        actions_value = value_function_q[curr_state[0], curr_state[1], :]
        return random_generator.choice(
            np.where(actions_value == np.max(actions_value))[0]
        )


def _compute_agent_policy(world: World, value_function_q: np.ndarray):
    policy = np.zeros(world.size, dtype=np.int8)

    for i in range(world.size[0]):
        for j in range(world.size[1]):
            policy[i, j] = np.argmax(value_function_q[i, j, :])

    return policy


def _test_policy(policy: np.ndarray, world: World,
                 agent: Agent, actions: List[Tuple[int, int]]):
    curr_state = agent.state
    max_steps = world.size[0] * world.size[1]

    for step in range(max_steps):
        if world.is_goal(curr_state):
            return True, step

        curr_action_id = policy[curr_state[0], curr_state[1]]
        curr_action = actions[curr_action_id]
        curr_state, _ = agent.check_action(curr_state, curr_action)

    return False, -1



