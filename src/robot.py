import random

class Robot:
    """
    Basic robot class that can move and keeps track of past postions
    """
    def __init__(self, start_pos, robot_search_radius=1):
        self.pos = start_pos
        self.path = [start_pos]
        self.recharging_procedure_time = 0
        self.mission_time = 0
        self.robot_search_radius = robot_search_radius
        self.battery = random.randint(90,100)

    def move(self, new_pos):
        if self.battery <= 0:
            return
        self.pos = new_pos
        self.path.append(self.pos)
        self.battery -= 0.75

    def get_position(self):
        return self.pos

    def get_path(self):
      return self.path

    def get_battery(self):
        return self.battery
