max_charge = 100  # total battery amount
distance_per_charge = 500  # distance that bot can move
battery_burn = (
    max_charge / distance_per_charge
)  # the amount of battery cost a movement has
import numpy as np


class Robot:
    """
    Basic robot class that can move and keeps track of past postions
    """

    def __init__(self, ID, start_pos):
        self.id = ID
        self.pos = start_pos
        self.path = [start_pos]
        self.recharging_procedure_time = 0
        self.mission_time = 0
        self.max_charge = max_charge
        self.battery = 100

    def move(self, new_pos):
        distance = np.linalg.norm(np.array(self.pos) - np.array(new_pos))
        if self.battery > 0:
            if self.battery - (battery_burn * distance) > 0:
                self.pos = new_pos
                self.path.append(self.pos)
                self.battery = self.battery - (battery_burn * distance)
            else:  # is there a better way to handle this
                self.battery = 0
                self.pos = new_pos
            return True
        return False

    def get_position(self):
        return self.pos

    def get_path(self):
        return self.path

    def get_id(self):
        return self.id

    def get_remaining_ratio(self):
        return self.battery / self.max_charge

    def get_near_known_charge(self):
        return self.path[0]


class Path_Node:
    def __init__(self, parent, pos, distance):
        self.parent = parent
        self.pos = pos
        self.distance = distance
        self.children = None

    def add_child(self, child):
        self.children.append(child)

    def get_pos(self):
        return self.pos

    def get_distance(self):
        return self.distance

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children
