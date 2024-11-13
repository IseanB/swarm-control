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
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-0.5, self.environment.get_size()[1] - 0.5)
        ax.set_ylim(-0.5, self.environment.get_size()[0] - 0.5)
        ax.set_xticks(np.arange(-0.5, self.environment.get_size()[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.environment.get_size()[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0)
        # Plot obstacles
        for obstacle in self.environment.get_obstacles():
            x, y = zip(*obstacle)
            ax.fill(x, y, color='blue', alpha=0.5)
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
        x_r = [bot.get_position()[0] for bot in self.swarm.actors]
        y_r = [bot.get_position()[1] for bot in self.swarm.actors]
        ax.scatter(x_r, y_r, marker='o', color='black', s=20, label='Robots')
        return fig, ax

    def save_paths(self, filename="path.png"):
        fig, ax = self.draw_map()
        for bot in self.swarm.actors:
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
        for bot in self.swarm.actors:
            path = bot.get_path()
            if len(path) > 1:
                x_pos, y_pos = zip(*path)
                ax.plot(x_pos, y_pos, linewidth=1)
        fig.suptitle("Swarm Paths")
        plt.legend(loc='upper right')
        plt.savefig(self.visualization_dir + 'frames/' + filename)
        plt.close()

    def animate_swarm(self, interval=200, filename='animation.gif'):
        """
        Animates the plotting of the swarm.
        """
        fig, ax = self.draw_map()
        lines = []
        # Create lines for each robot
        for bot in self.swarm.actors:
            line, = ax.plot([], [], lw=2)
            lines.append(line)

        # Animation function
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate_func(i):
            for line, bot in zip(lines, self.swarm.actors):
                path = bot.get_path()
                if i < len(path):
                    x_pos, y_pos = zip(*path[:i + 1])
                    line.set_data(x_pos, y_pos)
            return lines

        # Determine the number of frames
        max_steps = max(len(bot.get_path()) for bot in self.swarm.actors)

        anim = animation.FuncAnimation(
            fig, animate_func, init_func=init,
            frames=max_steps, interval=interval, blit=True
        )

        # Save the animation
        anim.save(self.visualization_dir+  filename, writer='ffmpeg')
        plt.close(fig)
