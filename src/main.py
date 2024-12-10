import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull
import time
import tqdm

from environment import *
from visualizer import *
from simulator import *
from wpt import *
from evaluate import *
from apf_exploration import *
from aoi import *

# Global param.

width = 100
height = 100
num_actors = 5
num_obstacles = 3
max_vertices = 4
max_size = 9
robot_search_radius = 1 # defined a circle around each robot that is considered "explored"
visualization_dir = "./visual_results/"
np.random.seed(1)

def run_ui(environment):
    flight_area_vertices = define_flight_area(environment)
    return flight_area_vertices

# APF Params
params = {
    "k_att": 30.0,
    "k_rep": 50.0,
    "k_bat": 5.0,
    "Q_star": 10.0,
    "delta": 1.0,
    "step_size": 0.5,
    "k_exp": 5.0,  # Coefficient for repulsion from explored areas
    "k_robot_rep": 1.0,  # Coefficient for repulsion between robots
    "robot_repulsion_distance": 1.0,  # Distance threshold for robot repulsion
    "explore_mult" : 1.0,
    "explore_threshold": 0.1,
    "obs_mult" : 1.0
}

np.random.seed(120) # Optional
rand_env = Environment((width, height))
rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
rand_env.add_survivors(5, (width/3, height/3), 10)


all_wpts = WPTS()
all_wpts.add_wpt((0,35),(0,0,0), math.pi / 5, initial_alpha=0)
all_wpts.add_wpt((0,60),(50,0,0), math.pi/2, initial_alpha=0)

simulator_2 = Simulator(num_actors, rand_env, all_wpts, init='wpt')

# Define parameters for the potential field

flight_area_vertices = run_ui(environment=rand_env)

# Create the potential field planner
potential_field = AdaptivePotentialField(rand_env, simulator_2, params, flight_area_vertices)

start_time = time.time()

dt = 1
# simulator_2.move_with_potential_field(potential_field, steps=200, search_range=robot_search_radius)
for i in tqdm(range(2000), "Running Simulation"):
    # simulator_2.basic_move_wpts(0.005)
    simulator_2.move_with_potential_field(potential_field,dt,robot_search_radius)
    simulator_2.autonomous_movement_wpts(omega=30, schedulingHz=4,step_dist=0.005)

    simulator_2.increment_a_clock()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Evaluation
evaluator = Evaluator(simulator_2, rand_env)
evaluator.evaluate()

# Visualizations
# gradient_plot(potential_field, [0,width], [0,height])
visualization = Visualizer(rand_env, simulator_2, potential_field)
visualization.save_occ_map(filename='occ_map_apf.png')  # Generates occ_map.png
visualization.save_paths(filename='paths_apf.png')    # Generates path.png
# visualization.plot_potential_field(potential_field, skip=5, filename="potential_field.png")
# visualization.animate_swarm(filename='animation_apf.gif')
