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

    def move_with_potential_field(self, potential_field, steps=100, search_range=1):
        for _ in range(steps):
            potential_field.compute_next_positions()
            self.detect_survivors(range=search_range)
            for bot in self.actors:
                self.environment.update_occ_map(bot.get_position(), search_range)
                bot.mission_time += 1
            for survivor in self.environment.get_survivors():
                survivor.increment_time()