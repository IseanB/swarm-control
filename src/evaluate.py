import numpy as np
import csv

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

    def evaluate_files(self, csv_file, text_file, run_id):
        # Sum of all `time_in_env` for each survivor
        total_time_in_env = sum(
            survivor.time_in_env for survivor in self.environment.get_survivors()
        )

        # Percentage of the occupancy map that has been searched (any cell with a value > 0)
        occ_map = self.environment.return_occ_map()
        searched_cells = np.count_nonzero(occ_map)
        total_cells = occ_map.size
        percentage_searched = (searched_cells / total_cells) * 100

        # Percentage of survivors found vs not found
        total_survivors = len(self.environment.get_survivors())
        survivors_found = sum(
            1 for survivor in self.environment.get_survivors() if survivor.is_found()
        )
        percentage_found = (survivors_found / total_survivors) * 100
        percentage_not_found = 100 - percentage_found

        # Total mission time and time spent in recharging procedure for each robot
        total_mission_time = sum(
            robot.mission_time for robot in self.swarm.actors.values()
        )
        total_recharges = sum(
            robot.num_recharges for robot in self.swarm.actors.values()
        )
        total_time_in_backtracking = sum(
            robot.time_in_backtrack for robot in self.swarm.actors.values()
        )
        percentage_recharging = (
            (total_time_in_backtracking / total_mission_time) * 100
            if total_mission_time
            else 0
        )

        # Ending Battery
        battery_vals = [robot.battery for robot in self.swarm.actors.values()]
        min_battery = min(battery_vals)
        max_battery = max(battery_vals)
        average_battery = sum(battery_vals) / len(self.swarm.actors)
        dead_bots = sum(1 if val == 0 else 0 for val in battery_vals)

        # Wpt Stats
        wpt_stats = [
            f"Bots Recharged for WPT: {wpt.bots_charged}"
            for wpt in self.swarm.wpts.wpts
        ]
        wpt_summary = "; ".join(wpt_stats)

        # Prepare data for CSV (single row for each run_id)
        csv_row = {
            "Run ID": run_id,
            "Total Time in Environment": total_time_in_env,
            "Percentage Searched": f"{percentage_searched:.2f}",
            "Percentage Found": f"{percentage_found:.2f}",
            "Percentage Not Found": f"{percentage_not_found:.2f}",
            "Total Mission Time": total_mission_time,
            "Number of Recharges": total_recharges,
            "Time in Backtracking": total_time_in_backtracking,
            "Percentage Recharging": f"{percentage_recharging:.2f}",
            "Min Battery": min_battery,
            "Max Battery": max_battery,
            "Average Battery": average_battery,
            "Dead Bots": dead_bots,
            "WPT Summary": wpt_summary,
        }

        # Write CSV row
        writer = csv.DictWriter(csv_file, fieldnames=csv_row.keys())
        if csv_file.tell() == 0:  # Write header only if the file is empty
            writer.writeheader()
        writer.writerow(csv_row)

        # Write to text file
        text_file.write("---------------------\n")
        text_file.write(f"Run ID: {run_id}\n")
        text_file.write("---------------------\n")
        text_file.write("Evaluation Metrics:\n")
        text_file.write("---------------------\n")
        for key, value in csv_row.items():
            text_file.write(f"{key}: {value}\n")
        text_file.write("---------------------\n")
        text_file.write("\n")
        text_file.write("\n")
