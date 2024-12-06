import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

rc("animation", html="jshtml")
from scipy.spatial import ConvexHull
import time
from tqdm import tqdm

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
num_actors = 4
num_obstacles = 3
max_vertices = 4
max_size = 9
robot_search_radius = (
    1  # defined a circle around each robot that is considered "explored"
)
visualization_dir = "./visual_results/"
np.random.seed(1)


def run_ui(environment):
    flight_area_vertices = define_flight_area(environment)
    return flight_area_vertices


def run_swarm_wpt():
    """ """
    map_width = 20
    map_height = 20
    num_actors = 2
    np.random.seed(1)
    env = Environment((map_width, map_height))

    # Obstacles
    obstacle_0 = [
        [25.0, 25.0],
        [27.5, 25.0],
        [30.0, 27.5],
        [30.0, 30.0],
        [27.5, 32.5],
        [25.0, 32.5],
        [22.5, 30.0],
        [22.5, 27.5],
    ]
    obstacle_0 = list(np.array(obstacle_0) / 2)
    obstacle_1 = [[37.5, 12.5], [40.0, 12.5], [40.0, 37.5], [37.5, 37.5]]
    obstacle_2 = [
        [12.5, 37.5],
        [15.0, 35.0],
        [17.5, 40.0],
        [18.75, 30.0],
    ]  # [[250, 750], [300, 700], [350, 800], [375, 600]]
    obstacles = [obstacle_0, obstacle_1, obstacle_2]
    env.set_obstacles(obstacles)

    # Survivors
    # env.add_survivors(5, (45, 25), 2)
    # env.add_survivors(5, (25, 25), 2)

    all_wpts = WPTS()
    all_wpts.add_wpt((0, 20), (0, 5, 0), 0, initial_alpha=0)

    params = {
        "k_att": 10.0,
        "k_rep": 10.0,
        "k_bat": 10.0,
        "Q_star": 10.0,
        "delta": 0.5,
        "step_size": 0.5,
        "k_exp": 1.0,  # Coefficient for repulsion from explored areas
        "k_robot_rep": 1.0,  # Coefficient for repulsion between robots
        "robot_repulsion_distance": 1.0,  # Distance threshold for robot repulsion
    }

    test_swarm = Simulator(
        num_actors,
        env,
        all_wpts,
        init="wpt",
    )

    # Evaluation
    evaluator = Evaluator(test_swarm, env)

    # Create the potential field planner

    # flight_area_vertices = run_ui(environment=env)
    flight_area_vertices = [[10, 17.5], [17.5, 17.5], [15, 10]]
    potential_field = AdaptivePotentialField(
        env, test_swarm, params, flight_area_vertices, animation_plot=True
    )
    start_time = time.time()
    # Move the swarm using the potential field
    print("Running Simulation... ")
    dt = 1
    for i in tqdm(range(100), "Running Simulation"):
        # simulator_2.basic_move_wpts(0.005)
        test_swarm.move_with_potential_field(potential_field, dt, robot_search_radius)
        test_swarm.autonomous_movement_wpts(omega=30, schedulingHz=2, step_dist=0.05)
        test_swarm.increment_a_clock()

        if 10 == test_swarm.survivors_found:
            print("All Survior(s) Found at time ", i, "!!!")
            break
        # print movement tree
        # test_swarm_apf.print_tree()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Evaluation
    evaluator = Evaluator(test_swarm, env)
    # evaluator.evaluate()

    # Visualizations
    # gradient_plot(potential_field, [0,width], [0,height])
    visualization = Visualizer(env, test_swarm, potential_field)
    visualization.save_occ_map(filename="wpt/occ_map_apf.png")  # Generates occ_map.png
    visualization.save_paths(filename="wpt/paths_apf.png")  # Generates path.png

    visualization.animate_swarm(filename="wpt/animation_apf.mp4", shortenDronePath=True)
    visualization.save_tree(filename="wpt/tree_apf.png")

    # evaluator.evaluate()
    # visualization.animate_apf(2)


# run_swarm_wpt()


def run_k_bat_alternator():
    map_width = 20
    map_height = 20
    num_actors = 1
    np.random.seed(1)

    # Obstacles
    obstacle_0 = [
        [25.0, 25.0],
        [27.5, 25.0],
        [30.0, 27.5],
        [30.0, 30.0],
        [27.5, 32.5],
        [25.0, 32.5],
        [22.5, 30.0],
        [22.5, 27.5],
    ]
    obstacle_0 = list(np.array(obstacle_0) / 2)
    obstacle_1 = [[37.5, 12.5], [40.0, 12.5], [40.0, 37.5], [37.5, 37.5]]
    obstacle_2 = [
        [12.5, 37.5],
        [15.0, 35.0],
        [17.5, 40.0],
        [18.75, 30.0],
    ]  # [[250, 750], [300, 700], [350, 800], [375, 600]]
    obstacles = [obstacle_0, obstacle_1, obstacle_2]

    for i in range(5):
        env = Environment((map_width, map_height))
        env.set_obstacles(obstacles)

        # Survivors
        # env.add_survivors(5, (45, 25), 2)
        # env.add_survivors(5, (25, 25), 2)

        all_wpts = WPTS()
        all_wpts.add_wpt((0, 20), (0, 5, 0), 0, initial_alpha=0)

        test_swarm = Simulator(
            num_actors,
            env,
            all_wpts,
            init="wpt",
        )
        print("Iteration: ", i)
        params = {
            "k_att": 10.0,
            "k_rep": 10.0,
            "k_bat": 10.0 * i,
            "Q_star": 10.0,
            "delta": 0.5,
            "step_size": 0.5,
            "k_exp": 1.0,  # Coefficient for repulsion from explored areas
            "k_robot_rep": 1.0,  # Coefficient for repulsion between robots
            "robot_repulsion_distance": 1.0,  # Distance threshold for robot repulsion
        }
        flight_area_vertices = [[10, 17.5], [17.5, 17.5], [15, 10]]
        potential_field = AdaptivePotentialField(
            env, test_swarm, params, flight_area_vertices, animation_plot=True
        )
        print("Running Simulation... ")
        dt = 1
        for i in tqdm(range(100), "Running Simulation"):
            # simulator_2.basic_move_wpts(0.005)
            test_swarm.move_with_potential_field(
                potential_field, dt, robot_search_radius
            )
            test_swarm.autonomous_movement_wpts(
                omega=30, schedulingHz=2, step_dist=0.05
            )
            test_swarm.increment_a_clock()

            if 10 == test_swarm.survivors_found:
                print("All Survior(s) Found at time ", i, "!!!")
                break
            # print movement tree
            # test_swarm_apf.print_tree()

        visualization = Visualizer(env, test_swarm, potential_field)
        visualization.save_paths(
            filename="k_bat/paths_apf_" + str(params["k_bat"]) + ".png"
        )  # Generates path.png
        visualization.animate_apf(
            0, filename=("k_bat/apf_arrows" + str(params["k_bat"]) + ".gif")
        )


def run_k_sim(param):
    map_width = 100
    map_height = 100
    num_actors = 6
    np.random.seed(1)
    env = Environment((map_width, map_height))
    # Obstacles
    obstacle_base = np.array(
        [
            [2, 1],
            [3, 1],
            [4, 2],
            [3, 3],
            [2, 3],
            [1, 2],
        ]
    )
    obstacle_scaled = obstacle_base * 5
    obstacle_0 = 1 * obstacle_scaled
    obstacle_1 = 1 * obstacle_scaled
    obstacle_2 = 1 * obstacle_scaled
    obstacle_3 = 1 * obstacle_scaled
    obstacle_4 = 1 * obstacle_scaled
    obstacle_5 = 1 * obstacle_scaled
    obstacle_6 = 1 * obstacle_scaled
    obstacle_7 = 1 * obstacle_scaled
    obstacle_8 = 1 * obstacle_scaled
    obstacle_9 = 1 * obstacle_scaled

    obstacle_0 += 75  # top right corner
    print(obstacle_0)
    obstacle_1 += 30  # middle
    print(obstacle_0)
    obstacle_2[:, 0] += 55  # over 55 up 50
    obstacle_2[:, 1] += 65
    obstacle_3[:, 1] += 50  # up 50
    obstacle_4[:, 1] += 65

    obstacle_5[:, 0] += 25
    obstacle_5[:, 1] += 57

    obstacle_6[:, 0] += 62
    obstacle_6[:, 1] += 35

    obstacles = [
        list(obstacle_0),
        list(obstacle_1),
        list(obstacle_2),
        list(obstacle_3),
        list(obstacle_4),
        list(obstacle_5),
        list(obstacle_6),
    ]
    env.set_obstacles(obstacles)

    for j in range(1):

        # Survivors
        env.add_survivors(5, (30, 90), 5)
        env.add_survivors(5, (90, 60), 5)
        # env.add_survivors(5, (25, 25), 2)
        print("Obstacles:", env.get_obstacles())
        all_wpts = WPTS()
        all_wpts.add_wpt((0, 100), (0, 5, 0), 0, initial_alpha=0)

        test_swarm = Simulator(
            num_actors,
            env,
            all_wpts,
            init="wpt",
        )

        print("Iteration: ", j)
        params = {
            "k_att": 10.0,
            "k_rep": 10.0,
            "k_bat": 10.0,
            "Q_star": 10.0,
            "delta": 0.5,
            "step_size": 0.5,
            "k_exp": 1.0,  # Coefficient for repulsion from explored areas
            "k_robot_rep": 1.0,  # Coefficient for repulsion between robots
            "robot_repulsion_distance": 1.0,  # Distance threshold for robot repulsion
        }
        params[param] = params[param] * (j + 1)
        flight_area_vertices = [[0, 40], [100, 40], [100, 100], [0, 100]]
        potential_field = AdaptivePotentialField(
            env, test_swarm, params, flight_area_vertices, animation_plot=False
        )
        print("Running Simulation... ")
        dt = 1
        for i in tqdm(range(100), "Running Simulation"):
            # simulator_2.basic_move_wpts(0.005)
            test_swarm.move_with_potential_field(
                potential_field, dt, robot_search_radius
            )
            test_swarm.autonomous_movement_wpts(
                omega=30, schedulingHz=2, step_dist=0.05
            )
            test_swarm.increment_a_clock()

            if 10 == test_swarm.survivors_found:
                print("All Survior(s) Found at time ", i, "!!!")
                break
            # print movement tree
            # test_swarm_apf.print_tree()

        visualization = Visualizer(env, test_swarm, potential_field)
        visualization.save_paths(
            filename="k_test/paths_apf_" + param + "_" + str(params[param]) + ".png"
        )  # Generates path.png
        visualization.save_occ_map(
            filename="k_test/occ_map_" + param + "_" + str(params[param]) + ".png"
        )  # Generates occ_map.png
        visualization.animate_swarm(
            filename="k_test/swarm_anim_" + param + "_" + str(params[param]) + ".mp4",
            shortenDronePath=True,
        )


run_k_sim("k_bat")
