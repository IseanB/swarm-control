import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull
import time

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

class Visualizer:
    """
    Handles all visualization-related functionalities.
    """
    def __init__(self, environment, swarm, visualization_dir=visualization_dir):
        self.environment = environment
        self.swarm = swarm
        self.visualization_dir = visualization_dir

    def display_occ_map(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.rot90(self.environment.return_occ_map()), cmap="PRGn", cbar=False)
        plt.suptitle("Occupied Map")
        plt.show()

    def save_occ_map(self, filename="occ_map.png"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.rot90(self.environment.return_occ_map()), cmap="PRGn", cbar=False)
        plt.title("Occupied Map")
        plt.savefig(self.visualization_dir + filename)
        plt.close()

    def draw_map(self,):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlim(-0.5, self.environment.get_size()[1] - 0.5)
        ax.set_ylim(-0.5, self.environment.get_size()[0] - 0.5)
        ax.set_xticks(np.arange(-0.5, self.environment.get_size()[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.environment.get_size()[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0)
        # Plot obstacles
        for obstacle in self.environment.get_obstacles():
            x, y = zip(*obstacle)
            ax.fill(x, y, color='grey', alpha=0.7)

        # Plot survivors
        survivors_found = [s for s in self.environment.get_survivors() if s.is_found()]
        survivors_not_found = [s for s in self.environment.get_survivors() if not s.is_found()]
        if survivors_not_found:
            x_nf, y_nf = zip(*[s.get_position() for s in survivors_not_found])
            ax.scatter(x_nf, y_nf, marker='*', color='red', s=100, label='Survivor Not Found')
        if survivors_found:
            x_f, y_f = zip(*[s.get_position() for s in survivors_found])
            ax.scatter(x_f, y_f, marker='*', color='green', s=100, label='Survivor Found')

        # Plot robots
        x_r = [bot.get_position()[0] for bot in self.swarm.actors.values()]
        y_r = [bot.get_position()[1] for bot in self.swarm.actors.values()]
        ax.scatter(x_r, y_r, marker='o', color='black', s=20, label='Robots')

        # Plot WPTs
        wpt_paths_data = [wpt.get_path() for wpt in self.swarm.wpts.get_wpts()]
        wpt_index = 0
        if(len(wpt_paths_data) != 0):
            for data in wpt_paths_data:
                wpt_index += 1

                x = [point[0] for point in data]
                y = [point[1] for point in data]

                ax.scatter(x, y, marker='^', s=10, alpha=0.7, label=f'WPT {wpt_index}')

        return fig, ax

    def save_paths(self, filename="path.png"):
        fig, ax = self.draw_map()
        for bot in self.swarm.actors.values():
            path = bot.get_path()
            if len(path) > 1:
                x_pos, y_pos = zip(*path)
                ax.plot(x_pos, y_pos, linewidth=1)
        fig.suptitle("Swarm Paths")
        plt.legend(loc='upper right')
        plt.savefig(self.visualization_dir + filename)
        plt.close()

    def save_frames(self, filename="path.png"):
        fig, ax = self.draw_map()
        for bot in self.swarm.actors.values():
            path = bot.get_path()
            if len(path) > 1:
                x_pos, y_pos = zip(*path)
                ax.plot(x_pos, y_pos, linewidth=1)
        fig.suptitle("Swarm Paths")
        plt.legend(loc='upper right')
        plt.savefig(self.visualization_dir + 'frames/' + filename)
        plt.close()

    def plot_apf_frame(self, frame, robot):
        fig, ax = self.draw_map()  # eventually move this up to the animation function
        X = []
        Y = []
        for r in range(self.environment.get_size()[0]):
            for c in range(self.environment.get_size()[1]):
                X.append(r)
                Y.append(c)  # same with this
        # print("X", X)

        self.plot_arrows(ax, frame, robot, X, Y)
        self.plot_robot(ax, frame, robot)
        fig.savefig(self.visualization_dir + "apf_frame.png")
        fig.clear()

    def plot_robot(self, ax, frame, robot):
        # print(robot.get_path())
        x = robot.get_path()[frame][0]
        y = robot.get_path()[frame][1]
        bot_point = ax.scatter(x, y, marker="o", color="blue", s=100, label="Robots")
        return bot_point

    def plot_arrows(self, ax, frame, robot, X, Y):
        U_vals = robot.get_arrow_directions()["U"][frame]
        # print(U_vals)
        V_vals = robot.get_arrow_directions()["V"][frame]
        Q = ax.quiver(X, Y, U_vals, V_vals, pivot="mid", angles="xy")
        return Q

    def animate_apf(self, robot_id, interval=200, filename="apf_arrows.gif"):
        """
        animates the apf for given robot.
        """
        fig, ax = self.draw_map()  # Initialize the figure and axes
        X = []
        Y = []
        for r in np.linspace(0, self.environment.get_size()[0], 25):
            for c in np.linspace(0, self.environment.get_size()[1], 25):
                X.append(r)
                Y.append(c)
        robot = self.swarm.actors[robot_id]
        # print(robot_id)

        quiver_obj = None
        robot_obj = None

        def update_plot(frame):
            nonlocal quiver_obj
            nonlocal robot_obj
            if quiver_obj:
                quiver_obj.remove()

            if robot_obj:
                robot_obj.remove()

            quiver_obj = self.plot_arrows(ax, frame, robot, X, Y)  # Update the arrows
            robot_obj = self.plot_robot(ax, frame, robot)  # Update the robot position
            return (fig,)

        # Assuming you have a way to determine the total number of frames
        num_frames = len(robot.get_arrow_directions()["U"])

        ani = animation.FuncAnimation(
            fig, update_plot, frames=num_frames, interval=interval, blit=True
        )

        ani.save(
            self.visualization_dir + str(robot_id) + "_" + filename, writer="pillow"
        )

    def animate_swarm(self, interval=200, filename='animation.gif', shortenDronePath=False, shortenWPTPath=True):
        """
        Animates the plotting of the swarm.
        """
        fig, ax = self.draw_map()
        lines = []

        # Create lines for each robot
        for bot in self.swarm.actors.values():
            line, = ax.plot([], [], linestyle='-', lw=2)
            lines.append(line)

        # Create lines for each wpt
        for _ in self.swarm.wpts.get_wpts():
            line, = ax.plot([], [], linestyle='-', lw=10, color='navy', alpha=0.7)
            lines.append(line)

        # Animation function
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate_func(i):
            for line, bot in zip(lines[:len(self.swarm.actors)], self.swarm.actors.values()): #animate robots
                path = bot.get_path()
                if(shortenDronePath):
                    if i < len(path) and i < 5:
                        x_pos, y_pos = zip(*path[:i+1])
                        line.set_data(x_pos, y_pos)
                    if i < len(path)-1 and i >= 5:
                        x_pos, y_pos = zip(*path[i-5:i+1])
                        line.set_data(x_pos, y_pos)
                else:
                    if i < len(path):
                        x_pos, y_pos = zip(*path[:i + 1])
                        line.set_data(x_pos, y_pos)

            for line, bot in zip(lines[len(self.swarm.actors):],self.swarm.wpts.get_wpts()): #animate robots
                path = bot.get_path()
                if(shortenWPTPath):
                    if i < len(path) and i < 2:
                        x_pos, y_pos = zip(*path[:i+1])
                        line.set_data(x_pos, y_pos)
                    if i < len(path)-1 and i >= 2:
                        x_pos, y_pos = zip(*path[i-2:i+1])
                        line.set_data(x_pos, y_pos)
                else:
                    if i < len(path):
                        x_pos, y_pos = zip(*path[:i + 1])
                        line.set_data(x_pos, y_pos)

            return lines

        # Determine the number of frames
        max_steps = max(len(bot.get_path()) for bot in self.swarm.actors.values())

        anim = animation.FuncAnimation(
            fig, animate_func, init_func=init,
            frames=max_steps, interval=interval, blit=True
        )

        # Save the animation
        anim.save(self.visualization_dir + filename, writer="pillow")
        plt.close(fig)
