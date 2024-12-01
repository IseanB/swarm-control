import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull
import time
from robot import *

# Global param.

width = 100
height = 100
num_actors = 20
num_obstacles = 15
max_vertices = 4
max_size = 100
visualization_dir = "./visual_results/"
np.random.seed(1)

class Simulator:
    """
    Swarm class that holds the robots, WPT, and environment
    contains methods to globally move the swarm around
    """
    def __init__(self, num_actors, environment, all_wpts, init):
      self.environment = environment
      self.num_actors = num_actors
      self.survivors_found = 0
      self.actors = []
      self.wpts = all_wpts
      self.wpts_loc = []
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
      # Cases: 1) out of bounds, 2) within another robot 3) within an obstacle
      if position[0] < 0 or position[1] < 0 or position[0] >= size[0] or position[1] >= size[1]:
        return False
      for bot in self.actors:
        if bot.get_position() == position:
          return False
      if self.environment.obstacle_collision(position):
        return False
      return True

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

    def schedule_WPT(self):
      self.wpts.scheduling(self.environment.return_occ_map(), self.actors, 82)
      return None

    def move_wpts(self, distance=0.05):
      self.wpts.move_wpts(distance)
      self.record_wpts_position()

    def record_wpts_position(self):
      locs = []
      for wpt in self.wpts.get_wpts():
        locs.append(wpt.get_pos())
      self.wpts_loc.append(locs)
    
    def get_wpts_position(self):
      return self.wpts_loc
    
    def print_wpts_pos(self):
      print(self.wpts_loc)
