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
from swarm import *
from apf_exploration import *


# Global param.

width = 15
height = 15
num_actors = 1
num_obstacles = 0
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

        # Ending Battery
        battery_vals = [robot.battery for robot in self.swarm.actors.values()]
        min_battery = min(battery_vals)
        max_battery = max(battery_vals)
        average_battery = sum(battery_vals) / num_actors
        dead_bots = sum(1 if val == 0 else 0 for val in battery_vals)
        dead_bot_list = []
        if dead_bots != 0:
            for robot in self.swarm.actors.values():
                if robot.battery == 0:
                    dead_bot_list.append(
                        "ID: "
                        + str(robot.get_id())
                        + " Position: "
                        + "("
                        + str(robot.get_position()[0])
                        + ", "
                        + str(robot.get_position()[1])
                        + ")"
                    )

        # Total mission time and time spent in recharging procedure for each robot
        total_mission_time = sum(
            robot.mission_time for robot in self.swarm.actors.values()
        )
        total_recharging_time = sum(
            robot.recharging_procedure_time for robot in self.swarm.actors.values()
        )
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
        print(
            "Min Battery: ",
            min_battery,
            ", Max Battery: ",
            max_battery,
            ", Average Battery: ",
            average_battery,
        )
        print("Number of Dead Bots: ", dead_bots)
        print(dead_bot_list)


### Random Walk
def run_swarm():
    np.random.seed(1)
    rand_env = Environment((width, height))
    rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
    rand_env.add_survivors(5, (width / 5, height / 2), 15)
    rand_env.add_survivors(10, (width / 2, height / 5), 20)

    test_swarm = Swarm(num_actors, rand_env, init="random")

    start_time = time.time()

    test_swarm.random_walk(15, robot_search_radius)
    test_swarm.print_tree()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Evaluation
    evaluator = Evaluator(test_swarm, rand_env)
    evaluator.evaluate()

    # Visualizations
    visualization = Visualizer(rand_env, test_swarm)
    visualization.save_occ_map(
        filename="occ_map_random_walk.png"
    )  # Generates occ_map.png
    visualization.save_paths(filename="path_random_walk.png")  # Generates path.png
    visualization.animate_swarm(
        filename="animation_random_walk.gif"
    )  # Generates animation.gif # this causes a lot of slowdowns

    # APF
    np.random.seed(1)
    rand_env = Environment((width, height))
    rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
    rand_env.add_survivors(5, (width / 5, height / 2), 15)
    rand_env.add_survivors(10, (width / 2, height / 5), 20)

    test_swarm = Swarm(num_actors, rand_env, init="random")

    # Define parameters for the potential field
    params = {
        "k_att": 100.0,  # Attractive potential gain
        "k_bat": 1.0,  # Attractive Battery Gain
        "k_rep": 1.0,  # Repulsive potential gain
        "Q_star": 10.0,  # Obstacle influence distance
        "step_size": 1.0,  # Step size for robot movement
        "delta": 1e-2,  # Small value for numerical gradient
    }

    # Create the potential field planner
    start_time = time.time()

    potential_field = AdaptivePotentialField(rand_env, test_swarm, params)

    # Move the swarm using the potential field
    test_swarm.move_with_potential_field(
        potential_field, steps=200, search_range=robot_search_radius
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Evaluation
    evaluator = Evaluator(test_swarm, rand_env)
    evaluator.evaluate()

    # Visualizations
    # gradient_plot(potential_field, [0,width], [0,height])
    visualization = Visualizer(rand_env, test_swarm)
    visualization.save_occ_map(filename="occ_map_apf.png")  # Generates occ_map.png
    visualization.save_paths(filename="paths_apf.png")  # Generates path.png
    # visualization.plot_potential_field(potential_field, skip=5, filename="potential_field.png")
    # visualization.animate_swarm(filename="animation_apf.gif")


def run_swarm_wpt():
    """ """
    map_width = 1000
    map_height = 1000
    np.random.seed(1)
    env = Environment((map_width, map_height))

    # Obstacles
    obstacle_0 = [
        [250, 250],
        [275, 250],
        [300, 275],
        [300, 300],
        [275, 325],
        [250, 325],
        [225, 300],
        [225, 275],
    ]
    obstacle_1 = [[750, 250], [800, 250], [800, 750], [750, 750]]
    obstacle_2 = [[250, 750], [300, 700], [350, 800], [375, 600]]
    obstacles = [obstacle_0, obstacle_1, obstacle_2]
    env.set_obstacles(obstacles)

    # Power Stations
    wpts = [[100, 100], [50, 500], [600, 900], [800, 200]]
    env.add_wpt_stations(wpts)

    # Survivors
    env.add_survivors(5, (900, 500), 15)
    env.add_survivors(5, (500, 500), 15)
    env.add_survivors(5, (300, 800), 15)

    test_swarm_rand = Swarm(num_actors, env, init="wpt")

    start_time = time.time()

    test_swarm_rand.random_walk(1000, robot_search_radius)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Evaluation
    evaluator = Evaluator(test_swarm_rand, env)
    # evaluator.evaluate()

    # Visualizations
    visualization = Visualizer(env, test_swarm_rand)
    visualization.save_occ_map(
        filename="wpt/occ_map_random_walk.png"
    )  # Generates occ_map.png
    visualization.save_paths(filename="wpt/path_random_walk.png")  # Generates path.png
    # visualization.animate_swarm(filename='swarm_wpt/animation_random_walk.gif') # Generates animation.gif # this causes a lot of slowdowns
    test_swarm_rand.print_tree()

    test_swarm_apf = Swarm(num_actors, env, init="wpt")
    # print(test_swarm_apf.actors)

    params = {
        "k_att": 100.0,  # Attractive potential gain
        "k_bat": 5,  # Attractive potential for charging
        "k_rep": 1.0,  # Repulsive potential gain
        "Q_star": 10.0,  # Obstacle influence distance
        "step_size": 1.0,  # Step size for robot movement
        "delta": 1e-2,  # Small value for numerical gradient
    }

    # Create the potential field planner
    start_time = time.time()

    potential_field = AdaptivePotentialField(env, test_swarm_apf, params)

    # Move the swarm using the potential field
    test_swarm_apf.move_with_potential_field(
        potential_field, steps=1000, search_range=robot_search_radius
    )

    # print movement tree
    # test_swarm_apf.print_tree()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Evaluation
    evaluator = Evaluator(test_swarm_apf, env)
    # evaluator.evaluate()

    # Visualizations
    # gradient_plot(potential_field, [0,width], [0,height])
    visualization = Visualizer(env, test_swarm_apf)
    visualization.save_occ_map(filename="wpt/occ_map_apf.png")  # Generates occ_map.png
    visualization.save_paths(filename="wpt/paths_apf.png")  # Generates path.png
    # visualization.plot_potential_field(potential_field, skip=5, filename="potential_field.png")
    # visualization.animate_swarm(filename="wpt/animation_apf.gif")


def run_swarm_tree_test():
    np.random.seed(1)
    rand_env = Environment((width, height))
    rand_env.random_obstacles(num_obstacles, max_vertices, max_size)
    rand_env.add_survivors(5, (width / 5, height / 2), 15)
    rand_env.add_survivors(10, (width / 2, height / 5), 20)

    test_swarm = Swarm(num_actors, rand_env, init="random")

    start_time = time.time()
    params = {
        "k_att": 100.0,  # Attractive potential gain
        "k_bat": 5,  # Attractive potential for charging
        "k_rep": 1.0,  # Repulsive potential gain
        "Q_star": 10.0,  # Obstacle influence distance
        "step_size": 1.0,  # Step size for robot movement
        "delta": 1e-2,  # Small value for numerical gradient
    }

    # Create the potential field planner
    start_time = time.time()

    potential_field = AdaptivePotentialField(rand_env, test_swarm, params)

    test_swarm.move_with_potential_field(
        potential_field, steps=200, search_range=robot_search_radius
    )
    test_swarm.print_tree()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    visualization = Visualizer(rand_env, test_swarm)
    visualization.save_paths(filename="path_random_walk_test.png")  # Generates path.png
    visualization.animate_swarm(
        filename="animation_random_walk_test.gif"
    )  # Generates animation.gif # this causes a lot of slowdowns


def test_apf_plotter():
    width = 50
    height = 50
    num_actors = 4
    num_obstacles = 1
    max_vertices = 4
    max_size = 5
    robot_search_radius = (
        1  # defined a circle around each robot that is considered "explored"
    )
    np.random.seed(1)
    rand_env = Environment((width, height))

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
    obstacle_1 = [[75.0, 25.0], [80.0, 25.0], [80.0, 75.0], [75.0, 75.0]]
    obstacle_2 = [[25.0, 75.0], [30.0, 70.0], [35.0, 80.0], [37.5, 60.0]]
    obstacles = [obstacle_0, obstacle_1, obstacle_2]
    rand_env.set_obstacles(obstacles)
    rand_env.add_survivors(5, (width / 5, height / 5), 15)

    test_swarm = Swarm(num_actors, rand_env, init="random")

    params = {
        "k_att": 100.0,  # Attractive potential gain
        "k_bat": 5,  # Attractive potential for charging
        "k_rep": 40.0,  # Repulsive potential gain
        "Q_star": 10.0,  # Obstacle influence distance
        "step_size": 1.0,  # Step size for robot movement
        "delta": 1e-2,  # Small value for numerical gradient
    }

    potential_field = AdaptivePotentialField(rand_env, test_swarm, params)
    test_swarm.move_with_potential_field(
        potential_field, steps=50, search_range=robot_search_radius
    )

    # print("U:", test_swarm.actors[0].get_arrow_directions()["U"][15])
    visualization = Visualizer(rand_env, test_swarm)
    visualization.animate_apf(0)
    visualization.animate_apf(1)
    visualization.animate_apf(2)
    visualization.animate_apf(3)
    visualization.animate_swarm(filename="apf_vis.gif")


test_apf_plotter()
# run_swarm_tree_test()
