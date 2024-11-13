import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.animation as animation
from matplotlib import rc
from matplotlib.path import Path
rc('animation', html='jshtml')
from scipy.spatial import ConvexHull
from visualizer import *

# Global parameters
width = 200
height = 200
num_actors = 20
num_obstacles = 15
max_vertices = 10
max_size = 10
robot_search_radius = 1  # Defined as a circle around each robot that is considered "explored"
visualization_dir = "./visual_results/"
np.random.seed(1)

class AdaptivePotentialField:
    """
    Implements a gradient planner using adaptive potential fields to guide the swarm to the survivors.
    """
    def __init__(self, environment, swarm, params):
        self.environment = environment
        self.swarm = swarm
        self.params = params  # Parameters for potential field
        self.visualizer = Visualizer(self.environment, self.swarm)
        self.step = 0

    def compute_attractive_potential(self, position):
        # Find the nearest survivor
        min_dist = np.inf
        nearest_survivor = None
        for survivor in self.environment.get_survivors():
            if not survivor.is_found():
                survivor_pos = survivor.get_position()
                dist = np.linalg.norm(np.array(position) - np.array(survivor_pos))
                if dist < min_dist:
                    min_dist = dist
                    nearest_survivor = survivor_pos

        # Compute the attractive potential based on the nearest survivor
        if nearest_survivor is not None:
            U_att = 0.5 * self.params['k_att'] * min_dist ** 2
        else:
            U_att = 0  # No survivors left to find

        return U_att, nearest_survivor  # Return the nearest survivor position as well

    def compute_repulsive_potential(self, position):
        U_rep = 0
        Q_star = self.params['Q_star']  # Obstacle influence distance
        for obstacle in self.environment.get_obstacles():
            dist = self.compute_distance_point_to_polygon(position, obstacle)
            if dist <= Q_star and dist != 0:
                U_rep += 0.5 * self.params['k_rep'] * (1.0 / dist - 1.0 / Q_star) ** 2
            else:
                U_rep += 0
        return U_rep

    def compute_distance_point_to_polygon(self, point, polygon):
        # Compute the minimum distance from point to the edges of the polygon
        min_dist = np.nan
        num_vertices = len(polygon)
        for i in range(num_vertices):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % num_vertices]
            dist = self.distance_point_to_segment(point, p1, p2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def distance_point_to_segment(self, point, p1, p2):
        # Compute the distance from point to the line segment p1-p2
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
            xx = x1
            yy = y1
        elif param > 1:
            xx = x2
            yy = y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D

        dx = x - xx
        dy = y - yy
        return np.sqrt(dx * dx + dy * dy)

    def compute_potential(self, position):
        U_att, _ = self.compute_attractive_potential(position)
        U_rep = self.compute_repulsive_potential(position)
        U = U_att + U_rep
        return U

    def compute_gradient(self, position):
        # Numerically approximate the gradient
        delta = self.params['delta']  # Small value for numerical gradient

        # Compute the attractive potential and nearest survivor
        U_att, nearest_survivor = self.compute_attractive_potential(position)
        if nearest_survivor is None:
            return np.zeros(2)  # No gradient if no survivors left

        # Compute gradient analytically for the attractive potential
        grad_att = self.params['k_att'] * (np.array(position) - np.array(nearest_survivor))

        # Compute repulsive gradient numerically
        grad_rep = np.zeros(2)
        pos_x_plus = [position[0] + delta, position[1]]
        pos_x_minus = [position[0] - delta, position[1]]
        pos_y_plus = [position[0], position[1] + delta]
        pos_y_minus = [position[0], position[1] - delta]
        U_rep_x_plus = self.compute_repulsive_potential(pos_x_plus)
        U_rep_x_minus = self.compute_repulsive_potential(pos_x_minus)
        U_rep_y_plus = self.compute_repulsive_potential(pos_y_plus)
        U_rep_y_minus = self.compute_repulsive_potential(pos_y_minus)
        grad_rep[0] = (U_rep_x_plus - U_rep_x_minus) / (2 * delta)
        grad_rep[1] = (U_rep_y_plus - U_rep_y_minus) / (2 * delta)

        # Total gradient is the sum of attractive and repulsive gradients
        grad = grad_att + grad_rep

        return grad

    def compute_next_positions(self):
        self.step += 1
        for robot in self.swarm.actors:
            current_pos = np.array(robot.get_position(), dtype=float)
            grad = self.compute_gradient(current_pos)
            if np.linalg.norm(grad) == 0:
                continue  # Skip if gradient is zero
            # Move in the negative gradient direction
            step_size = self.params['step_size']
            direction = -grad / np.linalg.norm(grad)
            new_pos = current_pos + step_size * direction
            new_pos = np.round(new_pos).astype(int)  # Assuming grid positions
            # Ensure new_pos is valid
            size = self.environment.get_size()
            if (
                0 <= new_pos[0] < size[0]
                and 0 <= new_pos[1] < size[1]
                and not self.environment.obstacle_collision(new_pos)
                and self.swarm.is_valid_move(tuple(new_pos))
            ):
                # Move robot
                robot.move(tuple(new_pos))
            # self.visualizer.save_frames(filename=f'paths_apf_{self.step}.png') 



def gradient_plot(potential_field, xlim, ylim, skip=10):
    """
    Plots the gradient field of the potential field over the environment.
    """
    # Create a grid over the environment
    x_min, x_max = xlim
    y_min, y_max = ylim
    x = np.arange(x_min, x_max, skip)
    y = np.arange(y_min, y_max, skip)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X, dtype=float)
    grad_U_x = np.zeros_like(U)
    grad_U_y = np.zeros_like(U)

    # Compute the potential and gradient at each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            position = (X[i, j], Y[i, j])
            if potential_field.environment.obstacle_collision(position):
                U[i, j] = np.nan  # Assign NaN to obstacles
                grad_U_x[i, j] = np.nan
                grad_U_y[i, j] = np.nan
                continue
            U[i, j] = potential_field.compute_potential(position)
            grad = potential_field.compute_gradient(position)
            grad_U_x[i, j] = -grad[0]  # Negative gradient
            grad_U_y[i, j] = -grad[1]

    # Plot the gradient field
    plt.figure(figsize=(12, 8))
    plt.title('Gradient Field of the Potential')
    Q = plt.quiver(X, Y, grad_U_x, grad_U_y, pivot='mid', units='inches')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{units}{s}$', labelpos='E', coordinates='figure')

    # Plot obstacles
    for obstacle in potential_field.environment.get_obstacles():
        x_obs, y_obs = zip(*obstacle)
        plt.fill(x_obs, y_obs, color='blue', alpha=0.5)

    # Plot survivors
    survivors_found = [s for s in potential_field.environment.get_survivors() if s.is_found()]
    survivors_not_found = [s for s in potential_field.environment.get_survivors() if not s.is_found()]
    if survivors_not_found:
        x_nf, y_nf = zip(*[s.get_position() for s in survivors_not_found])
        plt.scatter(x_nf, y_nf, marker='*', color='red', s=100, label='Survivor Not Found')
    if survivors_found:
        x_f, y_f = zip(*[s.get_position() for s in survivors_found])
        plt.scatter(x_f, y_f, marker='*', color='green', s=100, label='Survivor Found')

    # Plot robots' positions
    x_r = [bot.get_position()[0] for bot in potential_field.swarm.actors]
    y_r = [bot.get_position()[1] for bot in potential_field.swarm.actors]
    plt.scatter(x_r, y_r, marker='o', color='black', s=20, label='Robots')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()