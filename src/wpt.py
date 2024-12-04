import math
import numpy as np
import matplotlib.pyplot as plt

BASELINE_OMEGA = 30

class WPT:
    """
    Basic WPT class that can move.
    """
    def __init__(self, x_val, translation, rotation, initial_alpha=0):
        self.x_min = x_val[0]
        self.x_max = x_val[1]
        self.rotation = rotation
        self.translation = translation
        self.alpha = 0 #Range: [0,1]
        self.update_alpha(initial_alpha) # initial_alpha is initializing the wpt potition to be (initial_alpha)*100 percent into the path.
        self.pos = self.scene_transformation()
        self.path = [self.pos]

    def get_path(self):
        return self.path

    def get_pos(self):
        return self.pos

    def get_x(self, in_alpha=None):
        if in_alpha is not None:
            return self.x_min*(1-in_alpha) + self.x_max*(in_alpha)
        return self.x_min*(1-self.alpha) + self.x_max*(self.alpha)
    
    def update_alpha(self, new_alpha):
        if(new_alpha<0):
            self.alpha = 0
        elif(new_alpha>1):
            self.alpha = 1
        else:
            self.alpha = new_alpha
    
    def scene_transformation(self, amplitude=6,b=0.1, in_alpha=None):
        x = self.get_x(in_alpha)
        # In order to uncomment below, aka change paths to non linear function, requires refactoring of wpt path planning wr/ target drone. 
        # y = amplitude*math.sin(x*b) 
        y = x*b
        pos = np.array([x, y, 0])

        # Rotation matrix
        rot_mat = np.array([
            [math.cos(self.rotation), -math.sin(self.rotation), 0],
            [math.sin(self.rotation), math.cos(self.rotation), 0],
            [0, 0, 1]
        ])

        # Apply transformation
        transformed_pos = rot_mat @ pos + self.translation
        return transformed_pos[:2]

    def move(self, distance=.05):
        self.update_alpha(self.alpha + distance)
        self.pos = self.scene_transformation()
        self.path.append(self.pos)
        return self.pos

    def move_towards(self, curr_target, distance):
        if(self.alpha + distance > 1):
            self.move(-distance)
            return
        elif(self.alpha - distance < 0):
            self.move(distance)
            return

        forward_pos = self.scene_transformation(in_alpha=(self.alpha+distance))
        forward_dist = np.linalg.norm(np.array(forward_pos) - np.array(curr_target), ord=2)
        curr_dist = np.linalg.norm(np.array(self.pos) - np.array(curr_target), ord=2)

        if curr_dist > forward_dist:
            self.move(distance)
            # print("++++++++++++++++")
        else:
            self.move(-distance)
            # print("----------------")

class WPTS:
    def __init__(self):
        self.wpts = []
        self.assignments = [] # (wpt_index, drone_pos, "RECHARGE" or "FRONTIER")
        self.clock = 0

    def num_of_wpts(self):
        return len(self.wpts)

    def get_wpts(self):
        return self.wpts

    def add_wpt(self, x_val, translation, rotation, initial_alpha):
       self.wpts.append(WPT(x_val, translation, rotation, initial_alpha)) # represents state and ???

    def basic_move_wpts(self, distance=0.005): 
        for wpt in self.wpts:
            wpt.move(distance)  

    def scheduling(self, occupany_map, robots, omega=BASELINE_OMEGA):
        '''
        Input:
        - self.wpts: List of all WPTs in enviorment
        - occupany_map: global occupancy_map
        - robots: robots position w/ battery information
        - omega: critical battery level

        Output:
        - assignments: Points that each WPT needs to move toward.
        '''
        assignments = []
        robot_info = [(robot.get_position(), robot.get_battery()) for robot in robots]
        # print("robot_info", robot_info)

        # Select "Drones needing Recharge"(dr) with battery level less than omega and not dead.
        dr_pos = [robot[0] for robot in robot_info if (int(robot[1]) < int(omega) and robot[1] > 0)]
        
        # Select "FRontier"(fr) drones with battery level greater than omega.
        fr_pos = [robot[0] for robot in robot_info if robot[1] >= omega]
    
        # Calculate Distance/Proximity between WPTs and DR Drones
        distances_to_dr = []    
        if(len(dr_pos) != 0):
            distances_to_dr = []
            wpt_index = 0
            for wpt in self.wpts:
                for drone_pos in dr_pos:
                    distances = [np.linalg.norm(np.array(wpt.get_pos()) - np.array(drone_pos))]
                    distances_to_dr.append((distances[0], drone_pos, wpt_index))
                wpt_index += 1
            distances_to_dr.sort(key=lambda x: x[0]) # Sort by distance to prioritize closer drones

        # Calculate Distance/Proximity between WPTs and FR Drones
        distances_to_fr = []
        if(len(fr_pos) != 0):
            wpt_index = 0
            for wpt in self.wpts:
                for drone_pos in fr_pos:
                    distances = [np.linalg.norm(np.array(wpt.get_pos()) - np.array(drone_pos))]
                    distances_to_fr.append((distances[0], drone_pos, wpt_index))
                wpt_index += 1

            distances_to_fr.sort(key=lambda x: x[0]) # Sort by distance to prioritize closer drones

        # Assigning WPTs a drones to go towards
        assigned_wpts = [False] * len(self.wpts)

        for dist, drone_pos, wpt in distances_to_dr:
            if(not assigned_wpts[wpt] and drone_pos != 0):
                assignments.append((wpt, drone_pos, "RECHARGE"))
                assigned_wpts[wpt] = True
            if (assigned_wpts.count(True) == len(self.wpts)): # Bounds Complexity
                break
        
        for dist, drone_pos, wpt in distances_to_fr:
            if(not assigned_wpts[wpt]): 
                assignments.append((wpt, drone_pos, "FRONTIER"))
                assigned_wpts[wpt] = True
            if (assigned_wpts.count(True) == len(self.wpts)): # Bounds Complexity
                break

        assignments.sort(key=lambda x: x[0]) # Sort by wpt index
        # print("assignments", assignments)
        self.assignments = assignments
        return assignments


def plot_WPT_scene_transformations():
    # Define parameters
    x_range = (0, 10)
    translation = (2, 2, 0)
    rotation = math.pi / 4  # 30 degrees
    # translation = [0, 0, 0]
    # rotation = 0

    wpt = WPT(x_range, translation, rotation)

    # Generate points for varying alpha
    alphas = np.linspace(0, 1, 1000)
    transformed_positions = []
    for alpha in alphas:
        transformed_positions.append(wpt.move(alpha))

    # Extract x and y coordinates
    x_vals = [pos[0] for pos in transformed_positions]
    y_vals = [pos[1] for pos in transformed_positions]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label="Transformed Path")
    plt.scatter(translation[0], translation[1], color='red', label='Translation Point')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.title("Scene Transformations")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid()
    plt.show()

# plot_WPT_scene_transformations() # testing purposes