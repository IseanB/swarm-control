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


def run_k_sim():
    ## Set Up Map
    map_width = 50
    map_height = 50
    num_actors = 6

    # Obstacles
    obstacle_base = np.array(
        [
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [3.0, 3.0],
            [2.0, 3.0],
            [1.0, 2.0],
        ]
    )
    obstacle_scaled = obstacle_base * 2.5
    obstacle_0 = 1 * obstacle_scaled
    obstacle_1 = 1 * obstacle_scaled
    obstacle_2 = 1 * obstacle_scaled
    obstacle_3 = 1 * obstacle_scaled
    obstacle_4 = 1 * obstacle_scaled
    obstacle_5 = 1 * obstacle_scaled
    obstacle_6 = 1 * obstacle_scaled

    obstacle_0 += 75.0 / 2  # top right corner
    obstacle_1 += 30.0 / 2  # middle
    obstacle_2[:, 0] += 55.0 / 2  # over 55 up 50
    obstacle_2[:, 1] += 65.0 / 2
    obstacle_3[:, 1] += 50.0 / 2  # up 50
    obstacle_4[:, 1] += 65.0 / 2

    obstacle_5[:, 0] += 25.0 / 2
    obstacle_5[:, 1] += 57.0 / 2

    obstacle_6[:, 0] += 62.0 / 2
    obstacle_6[:, 1] += 35.0 / 2

    obstacles = [
        list(obstacle_0),
        list(obstacle_1),
        list(obstacle_2),
        list(obstacle_3),
        list(obstacle_4),
        list(obstacle_5),
        list(obstacle_6),
    ]

    # Define parameter ranges:
    """
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
            "exlpore_threshold": 0.1,
            "obs_mult" : 1.0
        }
    """
    # each is a list of possible values to try for the simulation
    # k_att_l = [1, 10, 20, 50, 100]  # the attraction of aoi
    # k_rep_l = [1, 10, 20, 50, 100]  # the repulsion of obstacles
    # k_bat_l = [1, 10, 20, 50, 100]  # how strong to shade towards battery
    # k_exp_l = [1, 10, 20, 50, 100]  # repulsion of already explored areas
    # Q_star_l = [1, 5, 10, 20]  # how far away do obstacles take effect
    # delta_l = [
    #     0.5,
    #     0.75,
    #     1,
    #     1.25,
    # ]  # how far to look when taking the numerical derivative of the potential
    # step_size_l = [1]  # does not change is a factor of the robot
    # k_robot_rep_l = [1, 10, 20, 50, 100]  # repulsion from robots
    # robot_repulsion_distance_l = [
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    # ]  # how far away that repulsion comes into effect
    # explore_mult_l = [1, 5, 10, 20]
    # explore_threshold_l = [0.1, 0.2, 0.03, 0.4]
    # obs_mult_l = [1, 5, 10, 20]

    k_att_l = [1, 10]  # the attraction of aoi
    k_rep_l = [1, 10]  # the repulsion of obstacles
    k_bat_l = [1, 10]  # how strong to shade towards battery
    k_exp_l = [1, 10]  # repulsion of already explored areas
    Q_star_l = [1, 5]  # how far away do obstacles take effect
    delta_l = [
        1
    ]  # how far to look when taking the numerical derivative of the potential
    step_size_l = [1]  # does not change is a factor of the robot
    k_robot_rep_l = [1, 10]  # repulsion from robots
    robot_repulsion_distance_l = [5]  # how far away that repulsion comes into effect
    explore_mult_l = [5]
    explore_threshold_l = [0.2]
    obs_mult_l = [5]

    # files to write to:

    csv_filename = visualization_dir + "k_test_test/evaluation_metrics.csv"
    text_filename = visualization_dir + "k_test_test/evaluation_metrics.txt"
    with open(csv_filename, mode="w", newline="") as csv_file, open(
        text_filename, mode="w"
    ) as text_file:
        pass  # Opening in "w" mode clears the file; no need to write anything here

    with open(csv_filename, mode="a", newline="") as csv_file, open(
        text_filename, mode="a"
    ) as text_file:

        for k_att in k_att_l:
            for k_rep in k_rep_l:
                for k_bat in k_bat_l:
                    for k_exp in k_exp_l:
                        for Q_star in Q_star_l:
                            for delta in delta_l:
                                for step_size in step_size_l:
                                    for k_robot_rep in k_robot_rep_l:
                                        for (
                                            robot_repulsion_distance
                                        ) in robot_repulsion_distance_l:
                                            for explore_mult in explore_mult_l:
                                                for (
                                                    explore_threshold
                                                ) in explore_threshold_l:
                                                    for obs_mult in obs_mult_l:

                                                        run_ID = (
                                                            str(k_att)
                                                            + "_"
                                                            + str(k_rep)
                                                            + "_"
                                                            + str(k_bat)
                                                            + "_"
                                                            + str(k_exp)
                                                            + "_"
                                                            + str(Q_star)
                                                            + "_"
                                                            + str(delta)
                                                            + "_"
                                                            + str(step_size)
                                                            + "_"
                                                            + str(k_robot_rep)
                                                            + "_"
                                                            + str(
                                                                robot_repulsion_distance
                                                            )
                                                            + "_"
                                                            + str(explore_mult)
                                                            + "_"
                                                            + str(explore_threshold)
                                                            + "_"
                                                            + str(obs_mult)
                                                        )
                                                        env = Environment(
                                                            (map_width, map_height)
                                                        )
                                                        env.set_obstacles(obstacles)

                                                        # Survivors
                                                        env.add_survivors(
                                                            5, (30 / 2, 90 / 2), 5
                                                        )
                                                        env.add_survivors(
                                                            5, (90 / 2, 60 / 2), 5
                                                        )
                                                        # env.add_survivors(5, (25, 25), 2)
                                                        all_wpts = WPTS()
                                                        all_wpts.add_wpt(
                                                            (0, 50),
                                                            (0, 5, 0),
                                                            0,
                                                            initial_alpha=0,
                                                        )

                                                        test_swarm = Simulator(
                                                            num_actors,
                                                            env,
                                                            all_wpts,
                                                            init="wpt",
                                                        )

                                                        params = {
                                                            "k_att": k_att,
                                                            "k_rep": k_rep,
                                                            "k_bat": k_bat,
                                                            "Q_star": Q_star,
                                                            "delta": delta,
                                                            "step_size": step_size,
                                                            "k_exp": k_exp,  # Coefficient for repulsion from explored areas
                                                            "k_robot_rep": k_robot_rep,  # Coefficient for repulsion between robots
                                                            "robot_repulsion_distance": robot_repulsion_distance,  # Distance threshold for robot repulsion
                                                            "explore_mult": explore_mult,
                                                            "explore_threshold": explore_threshold,
                                                            "obs_mult": obs_mult,
                                                        }

                                                        flight_area_vertices = [
                                                            [0, 40 / 2],
                                                            [100 / 2, 40 / 2],
                                                            [100 / 2, 100 / 2],
                                                            [0, 100 / 2],
                                                        ]
                                                        potential_field = (
                                                            AdaptivePotentialField(
                                                                env,
                                                                test_swarm,
                                                                params,
                                                                flight_area_vertices,
                                                                animation_plot=False,
                                                            )
                                                        )
                                                        dt = 1
                                                        for i in tqdm(
                                                            range(500),
                                                            "Running Simulation: "
                                                            + str(run_ID),
                                                        ):
                                                            # simulator_2.basic_move_wpts(0.005)
                                                            test_swarm.move_with_potential_field(
                                                                potential_field,
                                                                dt,
                                                                robot_search_radius,
                                                            )
                                                            test_swarm.autonomous_movement_wpts(
                                                                omega=30,
                                                                schedulingHz=2,
                                                                step_dist=0.05,
                                                            )
                                                            test_swarm.increment_a_clock()

                                                            if (
                                                                10
                                                                == test_swarm.survivors_found
                                                            ):
                                                                print(
                                                                    "All Survior(s) Found at time ",
                                                                    i,
                                                                    "!!!",
                                                                )
                                                                break
                                                            # print movement tree
                                                            # test_swarm_apf.print_tree()

                                                        visualization = Visualizer(
                                                            env,
                                                            test_swarm,
                                                            potential_field,
                                                        )
                                                        visualization.save_paths(
                                                            filename="k_test_test/paths_apf_"
                                                            + run_ID
                                                            + ".png"
                                                        )  # Generates path.png
                                                        visualization.save_occ_map(
                                                            filename="k_test_test/occ_map_"
                                                            + run_ID
                                                            + ".png"
                                                        )  # Generates occ_map.png

                                                        evaluator = Evaluator(
                                                            test_swarm, env
                                                        )
                                                        evaluator.evaluate_files(
                                                            csv_file=csv_file,
                                                            text_file=text_file,
                                                            run_id=run_ID,
                                                        )


if __name__ == "__main__":
    run_k_sim()
