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

### ---------- Random Walk ----------
# np.random.seed(10)
# rand_env = Environment((width, height))
# rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
# rand_env.add_survivors(5, (width/5, height/2), 15)
# rand_env.add_survivors(10, (width/2, height/5), 20)

# all_wpts = WPTS()
# # all_wpts.add_wpt((0,10),(2,2,0), math.pi / 4)

# simulator_1 = Simulator(num_actors, rand_env, all_wpts, init = 'random')

# start_time = time.time()
# simulator_1.random_walk(200,robot_search_radius)
# end_time = time.time()

# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")

# # Evaluation
# evaluator = Evaluator(simulator_1, rand_env)
# evaluator.evaluate()

# #Visualizations
# visualization = Visualizer(rand_env, simulator_1)
# visualization.save_occ_map(filename='occ_map_random_walk.png')  # Generates occ_map.png
# visualization.save_paths(filename='path_random_walk.png') # Generates path.png
# # visualization.animate_swarm(filename='animation_random_walk.gif') # Generates animation.gif # this causes a lot of slowdowns


def run_ui(environment):
    flight_area_vertices = define_flight_area(environment)
    return flight_area_vertices

def test_apf_plotter():
        width = 10
        height = 10
        num_actors = 4
        num_obstacles = 1
        max_vertices = 4
        max_size = 5
        np.random.seed(1)
        rand_env = Environment((width, height))

        rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
        rand_env.add_survivors(5, (width / 5, height / 5), 15)

        all_wpts = WPTS()

        flight_area_vertices = run_ui(environment=rand_env)

        # simulator_2 = Simulator(num_actors, rand_env, all_wpts, init='random')
        test_swarm = Simulator(num_actors, rand_env, all_wpts, init="random")

        params = {
            'k_att': 10.0, # 100.0
            'k_rep': 10.0,
            "k_bat": 5,  # Attractive potential for charging
            'Q_star': 10.0,
            'delta': 0.5, #  1e-2  
            'step_size': 0.5, # 1.0
            'k_exp': 1.0,  # Coefficient for repulsion from explored areas
            'k_robot_rep': 1.0,  # Coefficient for repulsion between robots
            'robot_repulsion_distance': 1.0,  # Distance threshold for robot repulsion
        }

        potential_field = AdaptivePotentialField(rand_env, test_swarm, params,flight_area_vertices)
        print("checkpoint 1")
        test_swarm.move_with_potential_field(
            potential_field, steps=10, search_range=robot_search_radius
        )
        print("checkpoint 2")
        # print("U:", test_swarm.actors[0].get_arrow_directions()["U"][15])
        visualization = Visualizer(rand_env, test_swarm)
        print("checkpoint 3")
        visualization.animate_apf(0)
        visualization.animate_apf(1)
        visualization.animate_apf(2)
        visualization.animate_apf(3)
        print("checkpoint 4")
        visualization.animate_swarm(filename="apf_vis.gif")
        print("checkpoint 5")

# APF Params
params = {
    'k_att': 10.0,
    'k_rep': 10.0,
    'Q_star': 10.0,
    'delta': 0.5,
    'step_size': 0.5,
    'k_exp': 1.0,  # Coefficient for repulsion from explored areas
    'k_robot_rep': 1.0,  # Coefficient for repulsion between robots
    'robot_repulsion_distance': 1.0,  # Distance threshold for robot repulsion
}

# ---------- APF Run ----------
np.random.seed(120)
rand_env = Environment((width, height))
rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
rand_env.add_survivors(5, (width/2, height/2), 200)
rand_env.add_survivors(10, (width/2, height/2), 400)

all_wpts = WPTS()
all_wpts.add_wpt((0,400),(10,40,0), math.pi / 4, initial_alpha=0.5)
all_wpts.add_wpt((0,600),(400,260,0), math.pi, initial_alpha=0.6)
# goal call scheduling
simulator_2 = Simulator(num_actors, rand_env, all_wpts, init='random')

# Define parameters for the potential field
params = {
    'k_att': 10.0, # 100.0
    'k_rep': 10.0,
    'Q_star': 10.0,
    'delta': 0.5, #  1e-2  
    'step_size': 0.5, # 1.0
    'k_exp': 1.0,  # Coefficient for repulsion from explored areas
    'k_robot_rep': 1.0,  # Coefficient for repulsion between robots
    'robot_repulsion_distance': 1.0,  # Distance threshold for robot repulsion
}


test_apf_plotter()

# flight_area_vertices = run_ui(environment=rand_env)

# # Create the potential field planner
# potential_field = AdaptivePotentialField(rand_env, simulator_2, params, flight_area_vertices)

# start_time = time.time()

# dt = 1
# # simulator_2.move_with_potential_field(potential_field, steps=200, search_range=robot_search_radius) 
# for i in range(50):
#     # simulator_2.basic_move_wpts(0.005)
#     simulator_2.move_with_potential_field(potential_field,dt,robot_search_radius)
#     simulator_2.autonomous_movement_wpts(omega=30, schedulingHz=4,step_dist=0.005)
    
#     simulator_2.increment_a_clock()

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")

# # Evaluation
# evaluator = Evaluator(simulator_2, rand_env)
# evaluator.evaluate()

# # Visualizations
# # gradient_plot(potential_field, [0,width], [0,height])
# visualization = Visualizer(rand_env, simulator_2)
# visualization.save_occ_map(filename='occ_map_apf.png')  # Generates occ_map.png
# visualization.save_paths(filename='paths_apf.png')    # Generates path.png
# # visualization.plot_potential_field(potential_field, skip=5, filename="potential_field.png")
# visualization.animate_swarm(filename='animation_apf.gif')