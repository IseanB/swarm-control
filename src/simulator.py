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
        # self.actors = [] # dict
        self.wpts = all_wpts  # added
        self.autonomous_scheduling_clock = 0  # added
        self.actors = {}
        self.global_explored_map = np.zeros(self.environment.get_size(), dtype=bool)
        next_pos = 0
        if init == "close":
            self.close_init()
        elif init == "random":
            self.random_init()
        elif init == "wpt":
            self.wpt_init()

    def increment_a_clock(self):
        self.autonomous_scheduling_clock += 1

    def is_valid_move(self, position):
        size = self.environment.get_size()
        # Cases: 1) out of bounds, 2) within another robot 3) within an obstacle
        if (
            position[0] < 0
            or position[1] < 0
            or position[0] >= size[0]
            or position[1] >= size[1]
        ):
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

    def autonomous_movement_wpts(self, omega, schedulingHz=10, step_dist=0.005): 
        all_wpts = self.wpts.get_wpts()
        if self.autonomous_scheduling_clock % schedulingHz == 0 or self.wpts.assignments == []: # reschedule targets for all wpts
            self.schedule_WPT(omega)

        if self.wpts.assignments != []: # ensuring no assignments = all drones dead
            if self.autonomous_scheduling_clock % schedulingHz != 0:
                for wpt_index in range(len(all_wpts)):  # move towards assigned drone
                    if wpt_index != self.wpts.assignments[wpt_index][0]:
                        throw("Error in autonomous_movement_wpts: index mismatch")
                    else:
                        curr_target = self.wpts.assignments[wpt_index][1]
                        all_wpts[wpt_index].move_towards(curr_target, step_dist)
        else:
            for wpt_index in range(len(all_wpts)):  # move towards assigned drone
                all_wpts[wpt_index].move(0)

    def schedule_WPT(self, omega):
        # print("self.actors", self.actors)
        self.wpts.scheduling(
            self.environment.return_occ_map(), self.actors.values(), omega
        )
        return None

    def basic_move_wpts(self, distance=0.05):
        self.wpts.basic_move_wpts(distance)

    # ------------------------------------------

    def update_global_explored_map(self):
        """
        Updates the global explored map by combining the local explored maps of all robots.
        """
        # Reset global map
        self.global_explored_map = np.zeros(self.environment.get_size(), dtype=bool)
        # Combine all robots' local maps
        for robot in self.actors.values():
            self.global_explored_map = np.logical_or(self.global_explored_map, robot.local_explored_map)

    def get_global_explored_map(self):
        return self.global_explored_map

    def close_init(self):
        next_pos = 0
        id = 0
        while len(self.actors) < self.num_actors:
            if self.is_valid_move((0, next_pos)):
                self.actors[id] = Robot(id, (0, next_pos), sensing_radius=1, detection_radius=1, swarm=self)
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
                self.actors[id] = Robot(id, pos, sensing_radius=1, detection_radius=1, swarm=self)
                id += 1

    def wpt_init(self):
        wpt = 0
        id = 0
        alpha = 0.0
        while len(self.actors) < self.num_actors:
            wpt_obj = self.wpts.get_wpts()[wpt]  # get position wpt
            pos = wpt_obj.scene_transformation(in_alpha=alpha)
            print(pos)
            self.actors[id] = Robot(
                id, pos, sensing_radius=1, detection_radius=1, swarm=self
            )
            wpt = (wpt + 1) % self.wpts.num_of_wpts()
            id += 1
            alpha += 1.0 / self.num_actors

        print(self.actors)

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
            if tuple(bot.get_position()) == tuple(position):
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
        for i in range(steps):

            potential_field.compute_next_positions()
            self.detect_survivors(range=search_range)
            for bot in self.actors.values():
                self.environment.update_occ_map(bot.get_position(), search_range)
                bot.mission_time += 1
            for survivor in self.environment.get_survivors():
                survivor.increment_time()

    def print_tree(self):
        root = self.actors[0].get_current_node().find_root()
        root.visualize_tree()
