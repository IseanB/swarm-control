import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull

# Global param.

width = 100
height = 100
num_actors = 10
num_obstacles = 10
max_vertices = 7
max_size = 10
visualization_dir = "visual_results/"


class Robot:
    """
    Basic robot class that can move and keeps track of past postions
    """
    def __init__(self, start_pos):
        self.pos = start_pos
        self.path = [start_pos]

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

    def update_occ_map(self, new_pos):
      # if(new_pos[0]<0 or new_pos[1]<0 or new_pos[0]>=self.size[0] or new_pos[1]>=self.size[0]):
      try:
        self.occupancy_map[new_pos] += 1
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



class Swarm:
    """
    Swarm class that holds the robots and environment
    contains methods to globally move the swarm around
    """
    def __init__(self, num_actors, environment, init):
      self.environment = environment
      self.num_actors = num_actors
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


    def random_walk(self, steps=100):
      for _ in range(steps):
        for bot in self.actors:
          current_pos = bot.get_position()
          new_pos = (current_pos[0] + np.random.randint(-1, 2), current_pos[1]+(np.random.randint(-1, 2)))
          #print(current_pos)
          #print(new_pos)
          #print("\n")
          if self.is_valid_move(new_pos):
            bot.move(new_pos)
            self.environment.update_occ_map(new_pos)


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


rand_env = Environment((width, height))
rand_env.random_obstacles(num_obstacles, max_vertices, max_size)

test_swarm = Swarm(num_actors, rand_env, init = 'random')
test_swarm.random_walk(100)
test_swarm.plot()
rand_env.save_occ_map()
test_swarm.animate() # this causes a lot of slowdowns