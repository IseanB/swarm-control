import math
import numpy as np
import matplotlib.pyplot as plt

class WPT:
    """
    Basic WPT class that can move.
    """
    def __init__(self, x_val, translation, rotation):
        self.x_min = x_val[0]
        self.x_max = x_val[1]
        self.rotation = rotation
        self.translation = translation
        self.alpha = 0
        self.pos = self.scene_transformation()
        self.path = [self.pos]

    def get_path(self):
        return self.path

    def get_pos(self):
        return self.pos

    def get_x(self):
        return self.x_min*(1-self.alpha) + self.x_max*(self.alpha)
    
    def update_alpha(self, new_alpha):
        if(new_alpha<0):
            self.alpha = 0
        elif(new_alpha>1):
            self.alpha = 1
        else:
            self.alpha = new_alpha
    
    def scene_transformation(self, amplitude=1,b=0.5):
        x = self.get_x()
        y = amplitude*math.sin(x*b)
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

class WPTS:
    def __init__(self):
        self.wpts = []

    def num_of_wpts(self):
        return len(self.wpts)

    def get_wpts(self):
        return self.wpts

    def add_wpt(self, x_val, translation, rotation):
       self.wpts.append(WPT(x_val, translation, rotation)) # represents state and ???

    def move_wpts(self, distance=0.05):   
        for wpt in self.wpts:
            wpt.move(distance)

    def scheduling(self, occupany_map, robots, omega=0.3):
        '''
        Input:
        - List of all WPTs in enviorment
        - global occupancy_map
        - robots position w/ battery information
        - omega, critical battery level

        Output:
        - WPTs movement toward assigned goal.
        '''
        PSelected = []

        #Store critical drones positions
        robot_info = [(robot.get_position(), robot.get_battery()) for robot in robots]

        # Filter out "Drones needing Recharge"(dr) with battery level less than omega
        dr_centroid_pos = [robot[0] for robot in robot_info if robot[1] < omega]

        # Filter out "FRontier"(fr) drones with battery level greater than omega
        fr_pos = [robot[0] for robot in robot_info if robot[1] >= omega]

        for wpt in self.wpts:
            # Calculate distance between WPT and DRones needing Recharge
            distance = [np.linalg.norm(np.array(wpt.get_pos()) - np.array(drone_pos)) for drone_pos in dr_centroid_pos]
            # Calculate distance between WPT and FRontier drones
            distance += [np.linalg.norm(np.array(wpt.get_pos()) - np.array(drone_pos)) for drone_pos in fr_pos]

            # Calculate PSelected
            PSelected.append(sum(distance))
        


        # print("Post filiter", dr_centroid_pos)


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