import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull
import time

from environment import *
from visualizer import *
from simulator import *
from apf import *
from wpt import *
from evaluate import *

# Global param.

width = 100
height = 100
num_actors = 20
num_obstacles = 10
max_vertices = 4
max_size = 100
robot_search_radius = 1 # defined a circle around each robot that is considered "explored"
visualization_dir = "./visual_results/"
np.random.seed(1)

### ---------- Random Walk ----------
np.random.seed(10)
rand_env = Environment((width, height))
rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
rand_env.add_survivors(5, (width/5, height/2), 15)
rand_env.add_survivors(10, (width/2, height/5), 20)

all_wpts = WPTS()
# all_wpts.add_wpt((0,10),(2,2,0), math.pi / 4)

simulator_1 = Simulator(num_actors, rand_env, all_wpts, init = 'random')

start_time = time.time()
simulator_1.random_walk(200,robot_search_radius)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Evaluation
evaluator = Evaluator(simulator_1, rand_env)
evaluator.evaluate()

#Visualizations
visualization = Visualizer(rand_env, simulator_1)
visualization.save_occ_map(filename='occ_map_random_walk.png')  # Generates occ_map.png
visualization.save_paths(filename='path_random_walk.png') # Generates path.png
# visualization.animate_swarm(filename='animation_random_walk.gif') # Generates animation.gif # this causes a lot of slowdowns

# ---------- APF Run ----------
np.random.seed(12)
rand_env = Environment((width, height))
rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
rand_env.add_survivors(5, (width/5, height/2), 15)
rand_env.add_survivors(10, (width/2, height/5), 20)

all_wpts = WPTS()
all_wpts.add_wpt((0,100),(20,50,0), math.pi / 4)
all_wpts.add_wpt((0,70),(60,10,0), math.pi)
# goal call scheduling
simulator_2 = Simulator(num_actors, rand_env, all_wpts, init='random')

# Define parameters for the potential field
params = {
    'k_att': 100.0,    # Attractive potential gain
    'k_rep': 1.0,  # Repulsive potential gain
    'Q_star': 10.0,  # Obstacle influence distance
    'step_size': 1.0, # Step size for robot movement
    'delta': 1e-2     # Small value for numerical gradient
}

# Create the potential field planner
potential_field = AdaptivePotentialField(rand_env, simulator_2, params)

start_time = time.time()

dt = 1
# simulator_2.move_with_potential_field(potential_field, steps=200, search_range=robot_search_radius)
for i in range(200):
    simulator_2.move_wpts(0.01)
    simulator_2.move_with_potential_field(potential_field,dt,robot_search_radius)
    


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Evaluation
evaluator = Evaluator(simulator_2, rand_env)
evaluator.evaluate()

# Visualizations
# gradient_plot(potential_field, [0,width], [0,height])
visualization = Visualizer(rand_env, simulator_2)
visualization.save_occ_map(filename='occ_map_apf.png')  # Generates occ_map.png
visualization.save_paths(filename='paths_apf.png')    # Generates path.png
# visualization.plot_potential_field(potential_field, skip=5, filename="potential_field.png")
# visualization.animate_swarm(filename='animation_apf.gif')