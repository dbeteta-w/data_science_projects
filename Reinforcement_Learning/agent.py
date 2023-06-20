import numpy as np


class Agent:

    def __init__(self, world, initial_state,
                 obstacle_reward, terminal_reward, free_celd_reward):
        # Crea un agente
        self.world = world
        self.state = np.array(initial_state)
        self.obstacle_reward = obstacle_reward
        self.terminal_reward = terminal_reward
        self.free_celd_reward = free_celd_reward

    def move(self, state, action):
        # Gestiona las transiciones de estados
        nextState = state + np.array(action)
        if nextState[0] < 0:
            nextState[0] = 0
        elif nextState[0] >= self.world.size[0]:
            nextState[0] = self.world.size[0] - 1
        if nextState[1] < 0:
            nextState[1] = 0
        elif nextState[1] >= self.world.size[1]:
            nextState[1] = self.world.size[1] - 1
        if self.world.map[(nextState[0], nextState[1])] == 2:
            aux = nextState
            for i in range(self.world.size[0]):
                for j in range(self.world.size[1]):
                    if self.world.map[(i, j)] == 2 and (
                            nextState[0] != i and nextState[1] != j):
                        aux = np.array([i, j])
            nextState = aux
        return nextState

    def reward(self, nextState):
        # Gestiona los refuerzos
        if self.world.map[(nextState[0], nextState[1])] == -1:
            # Refuerzo cuando el agente intenta moverse a un obstáculo
            reward = self.obstacle_reward
        elif self.world.map[(nextState[0], nextState[1])] == 1:
            # Refuerzo cuando el agente se mueve a una celda terminal
            reward = self.terminal_reward
        else:
            # Refuerzo cuando el agente se mueve a una celda libre
            reward = self.free_celd_reward
        return reward

    def check_action(self, state, action):
        # Planifica una acción
        nextState = self.move(state, action)
        if self.world.map[(state[0], state[1])] == -1:
            nextState = state
        reward = self.reward(nextState)
        return nextState, reward

    def execute_action(self, action):
        # Planifica y ejecuta una acción
        nextState = self.move(self.state, action)
        if self.world.map[(self.state[0], self.state[1])] == -1:
            nextState = self.state
        else:
            self.state = nextState
        reward = self.reward(nextState)
        return self.state, reward

    def is_terminal(self, state):
        return self.world.is_goal(state) or self.world.is_obstacle(state)