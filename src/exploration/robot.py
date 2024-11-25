import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

max_charge = 100  # total battery amount


distance_per_charge = 150  # distance that bot can move
battery_burn = (
    max_charge / distance_per_charge
)  # the amount of battery cost a movement has
# remaining distance = m
class Robot:
    """
    Basic robot class that can move and keeps track of past postions
    """

    def __init__(self, ID, start_pos):
        self.id = ID
        self.pos = start_pos
        self.path = [start_pos]
        self.recharging_procedure_time = 0
        self.mission_time = 0
        self.max_charge = max_charge
        self.battery = 100
        self.current_node = Path_Node(None, start_pos, 0)
        # Use this to do the arrow directions each bot keeps track of its historical arrow directions
        self.arrow_directions = {"U": [], "V": []}
        """
        self.arrow_directions{ 'U' = [[step1], [step2], [step3]]
        'V' = [[step1], [step2], [step3]]
        }
        self.arrow_directions['U'][step1] = [0, .5, .6, ...., .74, .25]
        """

    def move(self, new_pos):
        # print("moving to: ", new_pos)
        distance = np.linalg.norm(np.array(self.pos) - np.array(new_pos))
        if self.battery > 0:
            if self.battery - (battery_burn * distance) > 0:
                self.pos = new_pos
                self.path.append(self.pos)
                self.battery = self.battery - (battery_burn * distance)
            else:  # is there a better way to handle this
                self.battery = 0
                self.pos = new_pos
            return True
        return False

    def update_node(self, parent, pos):
        distance_moved = np.linalg.norm(np.array(parent.get_pos()) - np.array(pos))
        total_distance = distance_moved + parent.get_distance()
        new_node = Path_Node(parent, pos, total_distance)
        parent.add_child(new_node)
        self.current_node = new_node

    def set_node(self, node):
        self.current_node = node

    def get_position(self):
        return self.pos

    def get_path(self):
        return self.path

    def get_id(self):
        return self.id

    def get_remaining_ratio(self):
        return self.battery / self.max_charge

    def get_remaining_distance(self):
        return self.battery / battery_burn

    def get_near_known_charge(self):
        return self.path[0]

    def get_current_node(self):
        return self.current_node

    def add_arrow_directions(self, U, V):
        self.arrow_directions["U"].append(U)
        self.arrow_directions["V"].append(V)

    def get_arrow_directions(self):
        return self.arrow_directions


class Path_Node:
    def __init__(self, parent, pos, distance):
        self.parent = parent
        self.pos = pos
        self.distance = distance
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_pos(self):
        return self.pos

    def get_distance(self):
        return self.distance

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def find_root(self):
        # print(self.parent)
        # print(self)
        if self.get_parent() == None:
            # print("no parent")
            # print("returning ", self)
            return self
        else:
            # print("has parent")
            # print(self.parent)
            return self.parent.find_root()

    def depth_first_exists(self, pos):
        if self.pos == pos:
            # print("found, returning: ", self)
            return self  # Found the node

        for child in self.children:
            where_exists = child.depth_first_exists(pos)
            if where_exists:
                # print("found in child: ", where_exists)
                return (
                    where_exists  # Found the node in a child subtree return that node
                )
        return None  # Not found in this subtree

    def pos_visited(self, pos):
        root = self.find_root()
        # print("Root found", root)
        return root.depth_first_exists(pos)

    def visualize_tree(root):
        G = nx.DiGraph()

        def add_node_and_edges(node, parent=None):
            G.add_node(node.pos)
            if parent:
                G.add_edge(parent.pos, node.pos)
            for child in node.children:
                add_node_and_edges(child, node)

        add_node_and_edges(root)

        pos = nx.spring_layout(G)  # You can experiment with different layouts
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=20,
            node_color="skyblue",
            font_size=5,
            font_weight="bold",
            arrowsize=5,
        )
        plt.savefig("./visual_results/wpt/tree")

    def print_trees(self, level=0):
        """Prints the tree structure in a hierarchical format.

        Args:
            level (int, optional): Indentation level for the current node. Defaults to 0.
        """

        indent = "  " * level
        print(indent + "(" + str(self.pos[0]) + ", " + str(self.pos[1]) + ")")
        for child in self.children:
            child.print_trees(level + 1)
