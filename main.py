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

width = 200
height = 200
num_actors = 20
num_obstacles = 15
max_vertices = 10
max_size = 10
robot_search_radius = 1 # defined a circle around each robot that is considered "explored"
visualization_dir = "visual_results/"
np.random.seed(1)

class Survivor:
    """
    Represents a survivor in the environment. Survivors have positions and a status indicating
    whether they have been found by a robot.
    """
    def __init__(self, position):
        self.position = position
        self.status = "not found"
        self.time_in_env = 0

    def get_position(self):
        return self.position

    def is_found(self):
        return self.status == "found"

    def mark_as_found(self):
        self.status = "found"

    def increment_time(self):
        if(self.status != "found"):
          self.time_in_env += 1

class Robot:
    """
    Basic robot class that can move and keeps track of past postions
    """
    def __init__(self, start_pos):
        self.pos = start_pos
        self.path = [start_pos]
        self.recharging_procedure_time = 0
        self.mission_time = 0

    def move(self, new_pos):
        self.pos = new_pos
        self.path.append(self.pos)

    def get_position(self):
        return self.pos

    def get_path(self):
      return self.path

class Environment:
    """
    Holds the description of the environment such as size and obstacles
    V1: obstacles are each only a point
    V2: each obstacle is a list of vertices
    """
    def __init__(self, size=(8,8)):
      self.size = size
      self.obstacles = []
      self.survivors = []
      self.occupancy_map = np.zeros(size)

    def get_size(self):
      return self.size

    def get_obstacles(self):
      return self.obstacles

    def random_obstacles(self, num_obstacles, max_vertex, max_size):
      for _ in range(num_obstacles):
        self.obstacles.append(self.generate_random_polygon(num_vertices = np.random.randint(3, max_vertex+1), size = np.random.rand()*max_size, offsetx = np.random.rand()*self.size[0], offsety = np.random.rand()*self.size[1]))

    def generate_random_polygon(self, num_vertices, size, offsetx, offsety):
      """Generates a random non-intersecting polygon with the given number of vertices.
      ToDo add random offset for both x and y
      """
      # Generate random points
      points = np.random.rand(num_vertices, 2)
      #print(points)
      #print(size)
      points = np.multiply(points, size)
      #print(points)
      #print(offsetx)
      #print(offsety)
      points[:, 0] += offsetx
      points[:, 1] += offsety
      #print(points)
      # Compute the convex hull
      hull = ConvexHull(points)
      # Extract the vertices of the convex hull
      polygon_vertices = points[hull.vertices]
      #print(polygon_vertices)
      return polygon_vertices

    def set_obstacles(self, obstacles):
      self.obstacles = obstacles

    def update_occ_map(self, new_pos, radius):
      # if(new_pos[0]<0 or new_pos[1]<0 or new_pos[0]>=self.size[0] or new_pos[1]>=self.size[0]):
      try:
        x, y = new_pos
        size_x, size_y = self.size
        
        # Define bounds within the radius, ensuring they stay within the occupancy map limits
        x_min = max(0, x - radius)
        x_max = min(size_x, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(size_y, y + radius + 1)
        
        # Iterate over each cell within the defined bounds and update the occupancy map
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                # Check if the cell is within the specified radius from new_pos
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    self.occupancy_map[i, j] += 1
      except:
        print("ERROR: occupancy_map update FAILED ")
      #   print(new_pos)
      #   print(self.size)

    def return_occ_map(self):
      return (self.occupancy_map)
    
    def display_occ_map(self):
      plt.figure(figsize=(10, 8))
      sns.heatmap(np.rot90(self.occupancy_map), cmap="PRGn", cbar=False)
      plt.suptitle("Occupied Map")
      plt.show()
    
    def save_occ_map(self):
      plt.figure(figsize=(10, 8))
      sns.heatmap(np.rot90(self.occupancy_map), cmap="PRGn", cbar=False)
      plt.title("Occupied Map")
      plt.savefig(visualization_dir + "occ_map.png")

    def obstacle_collision(self, position):
      """
      Determines whether a point is inside an obstacle.
      """
      x, y = position
      inside = False

      for obstacle in self.obstacles:
        for i in range(len(obstacle)):
          x1, y1 = obstacle[i]
          x2, y2 = obstacle[(i+1) % len(obstacle)]
          if (y > min(y1, y2)) and (y <= max(y1, y2)) and (x <= max(x1, x2)):
            if y1 != y2:
              xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
              if xinters > x:
                inside = not inside
      return inside

    def add_survivors(self, num_survivors, centroid_point=(width/2, height/2), std_dev=1.0):
      for _ in range(num_survivors):
        while True:
          # Generate a position based on a Gaussian distribution around the centroid
          x = int(np.random.normal(centroid_point[0], std_dev))
          y = int(np.random.normal(centroid_point[1], std_dev))
          
          # Ensure the generated position is within the environment boundaries and not inside an obstacle
          if 0 <= x < self.size[0] and 0 <= y < self.size[1] and not self.obstacle_collision((x, y)):
            self.survivors.append(Survivor((x, y)))
            break
      
    def get_survivors(self):
        return self.survivors

class Swarm:
    """
    Swarm class that holds the robots and environment
    contains methods to globally move the swarm around
    """
    def __init__(self, num_actors, environment, init):
      self.environment = environment
      self.num_actors = num_actors
      self.survivors_found = 0
      self.actors = []
      next_pos = 0
      if init == 'close':
        self.close_init()
      elif init == 'random':
        self.random_init()

    def close_init(self):
      next_pos = 0
      while len(self.actors) < self.num_actors:
        if self.is_valid_move((0,next_pos)):
          self.actors.append(Robot((0,next_pos)))
        else:
          next_pos += 1

    def random_init(self):
      while len(self.actors) < self.num_actors:
        pos = (np.random.randint(0, self.environment.get_size()[0]), np.random.randint(0, self.environment.get_size()[1]))
        if self.is_valid_move(pos):
          self.actors.append(Robot(pos))

    def is_valid_move(self, position):
      size = self.environment.get_size()
      if self.obstacle_collision(position) or position[0] < 0 or position[1] < 0 or position[0] >= size[0] or position[1] >= size[1]:
        return False
      for bot in self.actors:
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
          x2, y2 = obstacle[(i+1) % len(obstacle)]
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
        for bot in self.actors:
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
        for bot in self.actors:
          current_pos = bot.get_position()
          new_pos = (current_pos[0] + np.random.randint(-1, 2), current_pos[1]+(np.random.randint(-1, 2)))
          #print(current_pos)
          #print(new_pos)
          #print("\n")
          if self.is_valid_move(new_pos):
            bot.move(new_pos)
            self.environment.update_occ_map(new_pos, search_range)
          bot.mission_time += 1
        self.detect_survivors(range = search_range)
        for survivor in self.environment.get_survivors():
          survivor.increment_time()

    def draw_map(self):
      fig, ax = plt.subplots()
      ax.set_xlim(-0.5, self.environment.get_size()[1]-0.5)
      ax.set_ylim(-0.5, self.environment.get_size()[0]-0.5)
      ax.set_xticks(np.arange(-0.5, self.environment.get_size()[1], 1), minor=True)
      ax.set_yticks(np.arange(-0.5, self.environment.get_size()[0], 1), minor=True)
      for obstacle in self.environment.get_obstacles():
        x, y = zip(*obstacle)
        plt.fill(x, y, color='blue', alpha=0.5)
      return fig, ax

    def plot(self):
      fig, ax = self.draw_map()
      for bot in self.actors:
        path = bot.get_path()
        x_pos = list(zip(*path))[0]
        y_pos = list(zip(*path))[1]
        #print(path)
        #print(x_pos)
        #print(y_pos)
        ax.plot(x_pos, y_pos)
      fig.suptitle("Swarm Paths")
      fig.savefig(visualization_dir + "path.png")


    def animate(self, interval=200, filename='animation.gif'):
      """
      Animates the plotting of the swarm.
      """
      fig, ax = self.draw_map()
      lines = []
      # Create lines for each data set
      for bot in self.actors:
          line, = ax.plot([], [], lw=2)
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
              line.set_data(x_pos[:i+1], y_pos[:i+1])
          return lines

      # Create the animation
      fig.suptitle("Animated Swarm Paths")
      anim = animation.FuncAnimation(fig, animate_func, init_func=init,
                                      frames=len(self.actors[0].get_path()), interval=interval, blit=True)

      # Save the animation
      anim.save(visualization_dir + filename, writer='ffmpeg')
      anim
      plt.close(fig)
  
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


class Evaluator:
    """
    Evaluates the performance of the simulation by calculating various metrics.
    """
    def __init__(self, swarm, environment):
        self.swarm = swarm
        self.environment = environment

    def evaluate(self):
        # Sum of all `time_in_env` for each survivor
        total_time_in_env = sum(survivor.time_in_env for survivor in self.environment.get_survivors())

        # Percentage of the occupancy map that has been searched (any cell with a value > 0)
        occ_map = self.environment.return_occ_map()
        searched_cells = np.count_nonzero(occ_map)
        total_cells = occ_map.size
        percentage_searched = (searched_cells / total_cells) * 100

        # Percentage of survivors found vs not found
        total_survivors = len(self.environment.get_survivors())
        survivors_found = sum(1 for survivor in self.environment.get_survivors() if survivor.is_found())
        percentage_found = (survivors_found / total_survivors) * 100
        percentage_not_found = 100 - percentage_found

        # Total mission time and time spent in recharging procedure for each robot
        total_mission_time = sum(robot.mission_time for robot in self.swarm.actors)
        total_recharging_time = sum(robot.recharging_procedure_time for robot in self.swarm.actors)
        percentage_recharging = (total_recharging_time / total_mission_time) * 100 if total_mission_time else 0

        # Display results
        print("---------------------")
        print("Evaluation Metrics:")
        print("---------------------")
        print(f"Total Time in Environment for All Survivors: {total_time_in_env}")
        print(f"Occupancy Map Percentage Searched: {percentage_searched:.2f}%")
        print(f"Percentage of Survivors Found: {percentage_found:.2f}%")
        print(f"Percentage of Survivors Not Found: {percentage_not_found:.2f}%")
        print(f"Time Spent in Recharging Procedure: {total_recharging_time}")
        print(f"Total Mission Time: {total_mission_time}")
        print(f"Percentage Time in Recharging Procedure: {percentage_recharging:.2f}%")


start_time = time.time()

### Main Code
rand_env = Environment((width, height))
rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
rand_env.add_survivors(5, (width/5, height/2), 15)
rand_env.add_survivors(10, (width/2, height/5), 20)

test_swarm = Swarm(num_actors, rand_env, init = 'random')
test_swarm.random_walk(200,robot_search_radius)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Evaluation
evaluator = Evaluator(test_swarm, rand_env)
evaluator.evaluate()

#Visualizations
visualization = Visualizer(rand_env, test_swarm)
visualization.save_occ_map()  # Generates occ_map.png
visualization.save_paths() # Generates path.png
# visualization.animate_swarm() # Generates animation.gif # this causes a lot of slowdowns