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