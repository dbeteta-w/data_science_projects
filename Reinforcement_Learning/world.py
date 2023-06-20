class World:

    def __init__(self, size, terminal, obstacle, hole):
        # Crea un mundo
        self.size = size
        self.map = {}

        for i in range(size[0]):
            for j in range(size[1]):
                # Estados libres
                self.map[(i, j)] = 0
                # Estados terminales
                for t in terminal:
                    if i == t[0] and j == t[1]:
                        self.map[(i, j)] = 1
                # Estados con obstáculos
                for o in obstacle:
                    if i == o[0] and j == o[1]:
                        self.map[(i, j)] = -1
                # Estados con agujeros de gusano
                for h in hole:
                    if i == h[0] and j == h[1]:
                        self.map[(i, j)] = 2

    def is_goal(self, state):
        return self.map[(state[0], state[1])] == 1

    def is_obstacle(self, state):
        return self.map[(state[0], state[1])] == -1
