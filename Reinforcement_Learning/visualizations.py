import numpy as np
import matplotlib.pyplot as plt
from world import *

def print_map(world):
    m = "["
    for i in range(world.size[0]):
        for j in range(world.size[1]):
            if world.map[(i, j)] == 0:
                m += " O "
            elif world.map[(i, j)] == -1:
                m += " X "
            elif world.map[(i, j)] == 1:
                m += " F "
            elif world.map[(i, j)] == 2:
                m += " T "
        if i == world.size[0] - 1:
            m += "]\n"
        else:
            m += "\n"
    print(m)


def print_policy(world, policy):
    p = "["
    for i in range(world.size[0]):
        for j in range(world.size[1]):
            if world.map[(i, j)] == 1:
                p += " \u2605"
            elif world.map[(i, j)] == -1:
                p += " X "
            elif world.map[(i, j)] == 2:
                p += " T "
            else:
                if policy[i][j] == 0:
                    p += " ^ "
                elif policy[i][j] == 1:
                    p += " V "
                elif policy[i][j] == 2:
                    p += " < "
                elif policy[i][j] == 3:
                    p += " > "
                else:
                    p += " x "
        if i == world.size[0] - 1:
            p += "]\n"
        else:
            p += "\n"
    print(p)


def show_policy(world_name, world, policy, converged, episodes, path_length):
    print(f"{world_name}")
    print_policy(world, policy)
    print(f"Converged: {converged}")
    print(f"Episodes needed to converge: {episodes}")
    print(f"Length of the path taken: {path_length}\n")


def plot_qmap(value_function_q):
    map = value_function_q.max(axis=2)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    cax = ax.matshow(map)
    fig.colorbar(cax)
    plt.show()


def plot_q_ev(policy, world, value_function_q, agent, actions, label=None):
    state = agent.state
    vec = []
    max_steps = world.size[0] * world.size[1]

    for step in range(max_steps):
        if world.is_goal(state):
            break

        action_idx = policy[state[0], state[1]]
        action = actions[action_idx]
        state, R = agent.check_action(state, action)
        vec.append(np.max(value_function_q[state[0], state[1], :]))

    plt.plot(vec, label=label)

