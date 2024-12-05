import numpy as np

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
        total_mission_time = sum(robot.mission_time for robot in self.swarm.actors.values())
        total_recharging_time = sum(
            robot.num_recharges for robot in self.swarm.actors.values()
        )
        percentage_recharging = (total_recharging_time / total_mission_time) * 100 if total_mission_time else 0

        # Ending Battery
        battery_vals = [robot.battery for robot in self.swarm.actors.values()]
        min_battery = min(battery_vals)
        max_battery = max(battery_vals)
        average_battery = sum(battery_vals) / len(self.swarm.actors)
        dead_bots = sum(1 if val == 0 else 0 for val in battery_vals)

        # Wpt Stats
        print_msg_wpt = ""
        for wpt in self.swarm.wpts.wpts:
            print_msg_wpt += "Bots Recharged for WPT: " + str(wpt.bots_charged) + " \n"

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
        print(print_msg_wpt)
