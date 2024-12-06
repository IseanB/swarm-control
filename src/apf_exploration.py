import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.animation as animation
from matplotlib import rc
from matplotlib.path import Path
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull
from visualizer import *
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

num_arrows = 25
class AdaptivePotentialField:
    """
    Implements a gradient planner using adaptive potential fields to guide the swarm to the survivors.
    """
    def __init__(self, environment, swarm, params, aoi_vertices, animation_plot=False):
        self.environment = environment
        self.swarm = swarm
        self.params = params  # Parameters for potential field
        self.step = 0
        self.assigned_survivors = {}  # Maps survivor positions to assigned robot IDs
        self.aoi_vertices = aoi_vertices
        self.aoi_polygon = Polygon(self.aoi_vertices)
        self.animation_plot = animation_plot

    def compute_attractive_potential(self, position, robot):
        """
        Computes the attractive potential for the robot. If the robot is assigned to a survivor,
        it is attracted to the survivor's position. Otherwise, it is attracted to unexplored areas.
        """

        # Compute the minimum distance to flight areas
        robot_point = Point(position)
        U0 = 1
        # No assigned survivor, attract to area of interest
        robot_point = Point(position)

        if self.aoi_polygon.contains(robot_point):
            # Inside the area of interest
            U_att = U0
        else:
            # Outside the area of interest, compute distance to polygon
            d = robot_point.distance(self.aoi_polygon)
            U_att = U0 + 0.5 * self.params["k_att"] * d**2

        # No assigned survivor, attract to unexplored areas
        unexplored_positions = np.argwhere(robot.local_explored_map == False)
        if len(unexplored_positions) == 0:
            # All areas have been explored
            return 0, None
        # Find the nearest unexplored position
        min_dist = np.inf
        for pos in unexplored_positions:
            dist = np.linalg.norm(np.array(position) - pos)
            if dist < min_dist:
                min_dist = dist
        U_att += 0.5 * self.params['k_att'] * min_dist ** 2
        battery_distance = np.linalg.norm(
            np.array(position) - np.array(robot.get_path()[0])
        )  # could modify to be the position of the nearest WPT
        U_att += (
            self.params["k_bat"]
            * 1
            / (robot.get_remaining_distance())
            * battery_distance
        )  # battery term
        return U_att

    def compute_repulsive_potential(self, position, robot):
        """
        Computes the repulsive potential at a given position due to obstacles,
        explored areas, and other robots.
        """
        U_rep = 0
        Q_star = self.params['Q_star']  # Obstacle influence distance
        dist_min = np.inf
        for y in range(self.environment.occupancy_map.shape[0]):
            for x in range(self.environment.occupancy_map.shape[0]):
                # Check if the cell is an obstacle
                if self.environment.occupancy_map[y, x] == -10:
                    # Calculate Euclidean distance
                    dist = np.sqrt((x - position[0])**2 + (y - position[1])**2)
                    dist_min = np.min([dist, dist_min])

        if dist_min < 1000:
            U_rep = 0.5 * self.params['k_rep'] * (1.0 / dist_min - 1.0 / Q_star) ** 2


        x, y = int(position[0]), int(position[1])
        half_width = 4 // 2

        # Get the bounds of the square region, ensuring they stay within the map limits
        x_min = max(x - half_width, 0)
        x_max = min(x + half_width + 1, self.environment.occupancy_map.shape[0])
        y_min = max(y - half_width, 0)
        y_max = min(y + half_width + 1, self.environment.occupancy_map.shape[1])

        # Extract the region from the explored_map
        region = self.environment.occupancy_map[x_min:x_max, y_min:y_max]

        # Sum the values in the region to get the total explored area
        total_explored = np.sum((region>0))
        U_rep += self.params['k_exp'] * total_explored

        # Repulsion from other robots
        for other_robot in self.swarm.actors.values():
            if other_robot.get_id() != robot.get_id():
                other_position = np.array(other_robot.get_position())
                dist = np.linalg.norm(position - other_position)
                U_rep += 0.5 * self.params['k_robot_rep'] * (1.0 / dist - 1.0 / self.params['robot_repulsion_distance']) ** 2
        return U_rep

    def compute_distance_point_to_polygon(self, point, polygon):
        """
        Computes the minimum distance from a point to the edges of a polygon (obstacle).
        """
        min_dist = np.inf
        num_vertices = len(polygon)
        for i in range(num_vertices):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % num_vertices]
            dist = self.distance_point_to_segment(point, p1, p2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def distance_point_to_segment(self, point, p1, p2):
        """
        Computes the distance from a point to a line segment defined by p1 and p2.
        """
        x, y = point
        x1, y1 = p1
        x2, y2 = p2
        A = x - x1
        B = y - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:
            param = dot / len_sq

        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D

        dx = x - xx
        dy = y - yy
        return np.sqrt(dx * dx + dy * dy)

    def compute_gradient(self, robot, position):
        """
        Computes the gradient of the potential field at the robot's position.
        """
        # position = np.array(robot.get_position(), dtype=float)
        delta = self.params['delta']

        grad_att = np.zeros(2)
        pos_x_plus = np.array([position[0] + delta, position[1]])
        pos_x_minus = np.array([position[0] - delta, position[1]])
        pos_y_plus = np.array([position[0], position[1] + delta])
        pos_y_minus = np.array([position[0], position[1] - delta])
        U_att_x_plus = self.compute_attractive_potential(pos_x_plus, robot)
        U_att_x_minus = self.compute_attractive_potential(pos_x_minus, robot)
        U_att_y_plus = self.compute_attractive_potential(pos_y_plus, robot)
        U_att_y_minus = self.compute_attractive_potential(pos_y_minus, robot)
        grad_att[0] = (U_att_x_plus - U_att_x_minus) / (2 * delta)
        grad_att[1] = (U_att_y_plus - U_att_y_minus) / (2 * delta)

        # Compute repulsive gradient numerically
        grad_rep = np.zeros(2)
        pos_x_plus = np.array([position[0] + delta, position[1]])
        pos_x_minus = np.array([position[0] - delta, position[1]])
        pos_y_plus = np.array([position[0], position[1] + delta])
        pos_y_minus = np.array([position[0], position[1] - delta])
        U_rep_x_plus = self.compute_repulsive_potential(pos_x_plus, robot)
        U_rep_x_minus = self.compute_repulsive_potential(pos_x_minus, robot)
        U_rep_y_plus = self.compute_repulsive_potential(pos_y_plus, robot)
        U_rep_y_minus = self.compute_repulsive_potential(pos_y_minus, robot)
        grad_rep[0] = (U_rep_x_plus - U_rep_x_minus) / (2 * delta)
        grad_rep[1] = (U_rep_y_plus - U_rep_y_minus) / (2 * delta)

        # Total gradient
        grad = grad_att + grad_rep

        return grad

    def compute_next_positions(self):
        """
        Computes and updates the next positions of all robots in the swarm.
        """
        self.step += 1

        for robot in self.swarm.actors.values():
            current_pos = np.array(robot.get_position(), dtype=float)

            # Update the robot's local explored map
            robot.update_local_explored_map()

        # Update the global explored map after all robots have updated their local maps
        self.swarm.update_global_explored_map()

        for robot in self.swarm.actors.values():
            current_pos = np.array(robot.get_position(), dtype=float)
            if (robot.get_current_node().get_distance() >= robot.get_remaining_distance()):  # turning back threshold
                if self.animation_plot:
                    U, V = self.get_arrow_directions(robot, True)
                if robot.get_current_node().get_parent():  # check if at root
                    self.swarm.move(
                        robot.get_id(), robot.get_current_node().get_parent().get_pos()
                    )  # move the current robot to its parent nodes position
            else:
                grad = self.compute_gradient(robot, current_pos)
                if self.animation_plot:
                    U, V = self.get_arrow_directions(robot, False)
                # Check for survivor detection within the robot's detection radius
                # --------------------------------------------- CHECK
                for survivor in self.environment.get_survivors():
                    if not survivor.is_found():
                        dist_to_survivor = np.linalg.norm(current_pos - np.array(survivor.get_position()))
                        if dist_to_survivor <= robot.detection_radius:
                            survivor.mark_as_found()
                            survivor.detected_by.add(robot.get_id())
                            # Assign the closest robot to the survivor if not already assigned
                            if survivor.get_position() not in self.assigned_survivors:
                                self.assign_closest_robot_to_survivor(survivor)
                            print(f"Survivor at {survivor.get_position()} detected by robot {robot.get_id()}")
                # --------------------------------------------- CHECK

                # Compute the gradient for movement
                if np.linalg.norm(grad) < 0.1:
                    grad = np.random.rand(2)  # Skip if gradient is zero

                # if np.linalg.norm(grad) == 0:
                #     print("zero gradient")
                #     self.swarm.move(robot.get_id(), tuple(robot.get_position()))
                #     robot.add_arrow_directions(U, V)
                #     continue

                # Move in the negative gradient direction
                step_size = self.params["step_size"]
                direction = -grad / np.linalg.norm(grad)
                new_pos = current_pos + step_size * direction
                # new_pos = np.round(new_pos).astype(int)  # Assuming grid positions

                # Ensure new_pos is valid
                size = self.environment.get_size()
                if (
                    0 <= new_pos[0] < size[0]
                    and 0 <= new_pos[1] < size[1]
                    and not self.environment.obstacle_collision(new_pos)
                    and self.swarm.is_valid_move(tuple(new_pos))
                ):
                    # Move robot
                    self.swarm.move(robot.get_id(), tuple(new_pos))
                # --------------------------------------------- CHECK - removed else
            if self.animation_plot:
                robot.add_arrow_directions(U, V)

    def get_arrow_directions(self, robot, backtracking):
        U = []
        V = []
        # want only 50 points evenly spaced
        for r in np.linspace(0, self.environment.get_size()[0], num_arrows):
            for c in np.linspace(0, self.environment.get_size()[1], num_arrows):
                if backtracking:  # if turning around no arrows
                    U.append(0)
                    V.append(0)
                else:  # moving forward means get arrow directions
                    grad = self.compute_gradient(robot, [r, c])
                    U.append(-1 * float(grad[0]))
                    V.append(-1 * float(grad[1]))

        return U, V

    def assign_closest_robot_to_survivor(self, survivor):
        """
        Assigns the closest robot to the survivor.
        """
        min_dist = np.inf
        closest_robot_id = None
        survivor_pos = np.array(survivor.get_position())

        for robot in self.swarm.actors.values():
            robot_pos = np.array(robot.get_position())
            dist = np.linalg.norm(robot_pos - survivor_pos)
            if dist < min_dist:
                min_dist = dist
                closest_robot_id = robot.get_id()

        # Assign the closest robot
        self.assigned_survivors[survivor.get_position()] = closest_robot_id
