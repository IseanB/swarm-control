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
from wpt import *
from evaluate import *

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