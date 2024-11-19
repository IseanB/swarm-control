import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull
import time
from robot import Robot


# Global param.

width = 100
height = 100
num_actors = 20
num_obstacles = 15
max_vertices = 4
max_size = 100
robot_search_radius = 1 # defined a circle around each robot that is considered "explored"
visualization_dir = "./visual_results/"
np.random.seed(1)


class Swarm:
    """
    Swarm class that holds the robots and environment
    contains methods to globally move the swarm around
    """
    def __init__(self, num_actors, environment, init):
        self.environment = environment
        self.num_actors = num_actors
        self.survivors_found = 0
        self.actors = {}
        if init == "close":
            self.close_init()
        elif init == "random":
            self.random_init()
        if init == "wpt":
            self.wpt_init()

    def close_init(self):
        next_pos = 0
        id = 0
        while len(self.actors) < self.num_actors:
            if self.is_valid_move((0, next_pos)):
                self.actors[id] = Robot(id, (0, next_pos))
                id += 1
            else:
                next_pos += 1

    def random_init(self):
        id = 0
        while len(self.actors) < self.num_actors:
            pos = (
                np.random.randint(0, self.environment.get_size()[0]),
                np.random.randint(0, self.environment.get_size()[1]),
            )
            if self.is_valid_move(pos):
                self.actors[id] = Robot(id, pos)
                id += 1

    def wpt_init(self):
        wpt_positions = self.environment.get_wpt_pos()
        wpt = 0
        id = 0
        while len(self.actors) < self.num_actors:
            pos = wpt_positions[wpt]
            self.actors[id] = Robot(id, pos)
            wpt = (wpt + 1) % len(wpt_positions)
            id += 1

    def move(self, id, new_pos):
        node = self.node_visited(new_pos)
        # print("Node: ", node)
        moved = self.actors[id].move(new_pos)
        if moved:
            if (
                node == None
            ):  # if not visited make a node with parent as the current node
                # print("correct")
                self.actors[id].update_node(self.actors[id].get_current_node(), new_pos)
            else:  # if it has been visited set the node to the visited node
                # print("incorrect")
                self.actors[id].set_node(node)

    def node_visited(self, pos):
        for robot in self.actors.values():
            node_visited = robot.get_current_node().pos_visited(pos)
            # print(robot.get_current_node())
            # print(node_visited)
            if node_visited:
                return node_visited
        return None

    def is_valid_move(self, position):
        size = self.environment.get_size()
        if (
            self.obstacle_collision(position)
            or position[0] < 0
            or position[1] < 0
            or position[0] >= size[0]
            or position[1] >= size[1]
        ):
            return False
        for bot in self.actors.values():
            if bot.get_position() == position:
                return False
        return True

    def obstacle_collision(self, position):
        """
        Determines whether a point is inside an obstacle.
        """
        x, y = position
        inside = False

        for obstacle in self.environment.get_obstacles():
            for i in range(len(obstacle)):
                x1, y1 = obstacle[i]
                x2, y2 = obstacle[(i + 1) % len(obstacle)]
                if (y > min(y1, y2)) and (y <= max(y1, y2)) and (x <= max(x1, x2)):
                    if y1 != y2:
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                        if xinters > x:
                            inside = not inside
        return inside

    def detect_survivors(self, range=1):
        """
        Checks if any robot is within a specified range of a survivor and marks them as found.
        """
        for bot in self.actors.values():
            bot_pos = bot.get_position()
            for survivor in self.environment.get_survivors():
                if not survivor.is_found():
                    survivor_pos = survivor.get_position()
                    distance = np.linalg.norm(np.array(bot_pos) - np.array(survivor_pos))
                    if distance <= range:
                        survivor.mark_as_found()
                        self.survivors_found += 1
                        # print(f"Survivor found at {survivor_pos} by robot at {bot_pos}.")

    def random_walk(self, steps=100, search_range=1):
        for _ in range(steps):
            for bot in self.actors.values():
                current_pos = bot.get_position()
                new_pos = (
                    current_pos[0] + np.random.randint(-1, 2),
                    current_pos[1] + (np.random.randint(-1, 2)),
                )
                # print(current_pos)
                # print(new_pos)
                # print("\n")
                if self.is_valid_move(new_pos):
                    self.move(bot.get_id(), new_pos)
                    self.environment.update_occ_map(new_pos, search_range)
                bot.mission_time += 1
            self.detect_survivors(range=search_range)
            for survivor in self.environment.get_survivors():
                survivor.increment_time()

    def move_with_potential_field(self, potential_field, steps=100, search_range=1):
        for _ in range(steps):
            potential_field.compute_next_positions()
            self.detect_survivors(range=search_range)
            for bot in self.actors.values():
                self.environment.update_occ_map(bot.get_position(), search_range)
                bot.mission_time += 1
            for survivor in self.environment.get_survivors():
                survivor.increment_time()

    def draw_map(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.environment.get_size()[1] - 0.5)
        ax.set_ylim(-0.5, self.environment.get_size()[0] - 0.5)
        ax.set_xticks(np.arange(-0.5, self.environment.get_size()[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.environment.get_size()[0], 1), minor=True)
        for obstacle in self.environment.get_obstacles():
            x, y = zip(*obstacle)
            plt.fill(x, y, color="blue", alpha=0.5)
        return fig, ax

    def plot(self):
        fig, ax = self.draw_map()
        for bot in self.actors.values():
            path = bot.get_path()
            x_pos = list(zip(*path))[0]
            y_pos = list(zip(*path))[1]
            # print(path)
            # print(x_pos)
            # print(y_pos)
            ax.plot(x_pos, y_pos)
        fig.suptitle("Swarm Paths")
        fig.savefig(visualization_dir + "path.png")

    def move_with_potential_field(self, potential_field, steps=100, search_range=1):
        for _ in range(steps):
            potential_field.compute_next_positions()
            self.detect_survivors(range=search_range)
            for bot in self.actors.values():
                self.environment.update_occ_map(bot.get_position(), search_range)
                bot.mission_time += 1
            for survivor in self.environment.get_survivors():
                survivor.increment_time()

    def animate(self, interval=200, filename='animation.gif'):
        """
        Animates the plotting of the swarm.
        """
        fig, ax = self.draw_map()
        lines = []
        # Create lines for each data set
        for bot in self.actors:
            (line,) = ax.plot([], [], lw=2)
            lines.append(line)

        # Animation function
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate_func(i):
            for line, bot in zip(lines, self.actors):
                path = bot.get_path()
                x_pos = list(zip(*path))[0]
                y_pos = list(zip(*path))[1]
                line.set_data(x_pos[: i + 1], y_pos[: i + 1])
            return lines

        # Create the animation
        fig.suptitle("Animated Swarm Paths")
        anim = animation.FuncAnimation(
            fig,
            animate_func,
            init_func=init,
            frames=len(self.actors[0].get_path()),
            interval=interval,
            blit=True,
        )

        # Save the animation
        anim.save(visualization_dir + filename, writer="ffmpeg")
        anim
        plt.close(fig)

    def print_tree(self):
        root = self.actors[0].get_current_node().find_root()
        root.visualize_tree()
