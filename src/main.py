import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull
import time
from rand_env import *
from visualizer import *
from swarm import *
from apf import *

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


### Random Walk
np.random.seed(1)
rand_env = Environment((width, height))
rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
rand_env.add_survivors(5, (width/5, height/2), 15)
rand_env.add_survivors(10, (width/2, height/5), 20)

test_swarm = Swarm(num_actors, rand_env, init = 'random')

start_time = time.time()

test_swarm.random_walk(200,robot_search_radius)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Evaluation
evaluator = Evaluator(test_swarm, rand_env)
evaluator.evaluate()

#Visualizations
visualization = Visualizer(rand_env, test_swarm)
visualization.save_occ_map(filename='occ_map_random_walk.png')  # Generates occ_map.png
visualization.save_paths(filename='path_random_walk.png') # Generates path.png
visualization.animate_swarm(filename='animation_random_walk.gif') # Generates animation.gif # this causes a lot of slowdowns

# APF
np.random.seed(1)
rand_env = Environment((width, height))
rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
rand_env.add_survivors(5, (width/5, height/2), 15)
rand_env.add_survivors(10, (width/2, height/5), 20)

test_swarm = Swarm(num_actors, rand_env, init='random')

# Define parameters for the potential field
params = {
    'k_att': 100.0,    # Attractive potential gain
    'k_rep': 1.0,  # Repulsive potential gain
    'Q_star': 10.0,  # Obstacle influence distance
    'step_size': 1.0, # Step size for robot movement
    'delta': 1e-2     # Small value for numerical gradient
}

# Create the potential field planner
start_time = time.time()

potential_field = AdaptivePotentialField(rand_env, test_swarm, params)

# Move the swarm using the potential field
test_swarm.move_with_potential_field(potential_field, steps=200, search_range=robot_search_radius)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Evaluation
evaluator = Evaluator(test_swarm, rand_env)
evaluator.evaluate()

# Visualizations
# gradient_plot(potential_field, [0,width], [0,height])
visualization = Visualizer(rand_env, test_swarm)
visualization.save_occ_map(filename='occ_map_apf.png')  # Generates occ_map.png
visualization.save_paths(filename='paths_apf.png')    # Generates path.png
# visualization.plot_potential_field(potential_field, skip=5, filename="potential_field.png")
visualization.animate_swarm(filename='animation_apf.gif')