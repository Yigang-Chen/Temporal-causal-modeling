import networkx as nx
from numpy import *
import matplotlib.pyplot as plt
import hashlib
import concurrent.futures
import math


class Node:
    def __init__(self, name):
        """ Initialize a node with a name and empty lists for incoming and outgoing edges """
        self.name = name
        self.in_edges = []  # List to hold all incoming edges
        self.out_edges = []  # List to hold all outgoing edges

class Edge:
    def __init__(self, from_node, to_node, effect, score, attribute=None):
        """ Initialize an edge with a from_node, to_node, effect, and attribute"""
        self.from_node = from_node  # The node from which this edge start
        self.to_node = to_node  # The node to which this edge end

        self.effect = effect  # The effect of this edge
        self.score = score  # Additional IF score attribute 
        self.attribute = attribute  # Additional attributes for this edge

        # Adding this edge to the nodes' edge lists
        self.from_node.out_edges.append(self)
        self.to_node.in_edges.append(self)

class Network:
    def __init__(self):
        self.nodes = {}  # Nodes are stored in a dictionary for quick access by node name

    def add_node(self, name):
        """ Add a node to the network if it does not already exist
        Input: name (string): The name of the node to add.
        Output: None. Adds a new node with the given name to the network if it doesn't already exist.
        """
        if name not in self.nodes:
            self.nodes[name] = Node(name)

    def remove_node(self, name):
        """ Remove a node from the network 
        Input: name (string): The name of the node to remove.
        Output: None. Removes the node with the given name from the network if it exists."""
        if name in self.nodes: # Only remove the node if it exists in the network
            node = self.nodes[name]
            # Remove all incoming and outgoing edges of the node
            while node.in_edges:
                self.remove_edge(node.in_edges[0].from_node.name, name)
            while node.out_edges:
                self.remove_edge(name, node.out_edges[0].to_node.name)
            # Finally, remove the node itself
            del self.nodes[name]

    def add_edge(self, from_node_name, to_node_name, effect, score, attribute="V", virtual_flag = False):
        """ Add an edge to the network 
        Input: from_node_name (string): The name of the node from which the edge starts.
               to_node_name (string): The name of the node to which the edge ends.
               effect (int): The effect of the edge.
               score (int): The score of the edge.
               attribute (string): The attribute of the edge.
        Output: None. Adds a new edge to the network if it doesn't already exist."""
        # Handle compound nodes (those containing semicolons) For other databases, custmoization is needed
        from_node_name, to_node_name = str(from_node_name), str(to_node_name)

        if from_node_name != to_node_name:
            if not virtual_flag:
                from_node_names = from_node_name.split(";")
                to_node_names = to_node_name.split(";")

                if len(from_node_names) > 1 or len(to_node_names) > 1:
                    # Create individual sub-nodes and add virtual edges
                    for node_name in from_node_names:
                        self.add_node(node_name)
                        if node_name != from_node_name:
                            self.add_edge(node_name, from_node_name, 5, 1, virtual_flag=True)  # virtual edge
                    for node_name in to_node_names:
                        self.add_node(node_name)
                        if node_name != to_node_name:
                            self.add_edge(to_node_name, node_name, 5, 1, virtual_flag=True)  # virtual edge

            # Add the main edge
            if from_node_name not in self.nodes:
                self.add_node(from_node_name)
            if to_node_name not in self.nodes:
                self.add_node(to_node_name)

            from_node = self.nodes[from_node_name]
            to_node = self.nodes[to_node_name]
            
            # Check if the edge already exists
            for edge in from_node.out_edges:
                if edge.to_node == to_node and edge.attribute == attribute:
                    return  # If such an edge exists, we skip the addition

            # If no such edge exists, we create a new one
            Edge(from_node, to_node, effect, score, attribute)

    def remove_edge(self, from_node_name, to_node_name):
        """ Remove an edge from the network 
        Input: from_node_name (string): The name of the node from which the edge starts.
               to_node_name (string): The name of the node to which the edge ends.
        Output: None. Removes the edge between the two nodes if it exists."""
        # Check if both nodes exist in the network before removing an edge
        if from_node_name in self.nodes and to_node_name in self.nodes:
            from_node = self.nodes[from_node_name]
            to_node = self.nodes[to_node_name]
            # Search for the edge in the from_node's out_edges list
            for edge in from_node.out_edges:
                if edge.to_node is to_node:
                    # If the edge is found, remove it from both nodes' edges lists
                    from_node.out_edges.remove(edge)
                    to_node.in_edges.remove(edge)
                    return  # We assume there is only one edge between two nodes
        
        #     # Raise an error if the edge does not exist
        #     raise ValueError('Edge does not exist')
        # else:
        #     # Raise an error if the nodes don't exist
        #     raise ValueError('Both nodes need to exist before removing an edge')
    
    def has_edge(self, from_node, to_node):
        """ Check if an edge exists between two nodes 
        Input: from_node (string): The name of the node from which the edge starts.
               to_node (string): The name of the node to which the edge ends.
        Output: True if an edge exists between the two nodes, False otherwise."""
        if from_node in self.nodes and to_node in self.nodes:
            for edge in self.nodes[from_node].out_edges:
                if edge.to_node.name == to_node:
                    return True
        return False
    
    def get_nodes_edges(self):
        """ Retrieve the network data as lists of nodes and edges.
        Output:
            nodes (list): A list of node names.
            edges (list): A list of tuples, each containing (from_node, to_node, effect, score, attribute) for each edge.
        """
        nodes = list(self.nodes.keys())
        edges = []
        for node_name in nodes:
            node = self.nodes[node_name]
            for edge in node.out_edges:
                edges.append((edge.from_node.name, edge.to_node.name, edge.effect, edge.score, edge.attribute))

        return nodes, edges

    def intersect_with(self, other_network):
        """ Intersect the network with another network 
        Input: other_network (Network): The network to intersect with.
        Output: intersected_network (Network): The intersected network."""
        intersected_network = Network()  # Create new network class to hold the intersected network
        for node_name, node in self.nodes.items():
            if node_name in other_network.nodes:  # Check if the node exists in the other network
                for edge in node.out_edges:
                    # Check for an equivalent edge in the other network
                    equivalent_edge_exists = any(
                        edge.to_node.name == other_edge.to_node.name and
                        edge.effect == other_edge.effect
                        for other_edge in other_network.nodes[node_name].out_edges
                    )
                    if equivalent_edge_exists:
                        # Add the edge to the intersected network
                        intersected_network.add_edge(node_name, edge.to_node.name, edge.effect, edge.score, edge.attribute)
        return intersected_network

    def remove_degree_one_nodes(self, start_nodes, deg_up, deg_down):
        """ Remove degree one nodes from the network"""
        # Repeat until all degree one nodes are removed
        special_end_nodes = deg_up + deg_down
        while True:
            # Keep track of whether a node was removed in this iteration
            removed_node = False

            # Create a list of nodes to remove to avoid list change during iteration
            nodes_to_remove = []

            for node_name, node in self.nodes.items():
                # For special start nodes, only count in-edges
                if node_name in start_nodes and len(node.in_edges) == 0 and len(node.out_edges) > 0:
                    continue
                # For special end nodes, only count out-edges
                elif node_name in special_end_nodes and len(node.out_edges) == 0 and len(node.in_edges) > 0:
                    continue
                # For all other nodes, count both in- and out-edges
                elif len(node.in_edges) + len(node.out_edges) == 1:
                    nodes_to_remove.append(node_name)
                    removed_node = True

            # Remove nodes marked for removal
            for node_name in nodes_to_remove:
                self.remove_node(node_name)

            # If no node was removed in this iteration, break the loop
            if not removed_node:
                break
    
    def multi_process_bfs(self, start_end_pairs, cutoff):
        """Perform BFS for multiple start-end node pairs using multiple processes
        Input: start_end_pairs (list): A list of tuples where each tuple contains the start and end node names.
               cutoff (int): The maximum path length to consider.
        Output: A list of lists of paths between each start and end node pair."""
        # start_end_pairs is a list of tuples, where each tuple is (start_node_name, end_node_name)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.bfs_all_paths, (start_node_name, end_node_name, cutoff)) for start_node_name, end_node_name in start_end_pairs]            
        return [f.result() for f in futures]

    def bfs_all_paths(self, start_node_name, end_node_name, cut_off):
        """Find all paths between two nodes using BFS
        Input: start_node_name (string): The name of the start node.
               end_node_name (string): The name of the end node.
               cut_off (int): The maximum path length to consider.
        Output: A list of paths between the start and end nodes. Each path is a list of node names."""
        if start_node_name not in self.nodes or end_node_name not in self.nodes:
            raise ValueError('Both nodes need to exist in the network')

        start_node = self.nodes[start_node_name]
        end_node = self.nodes[end_node_name]
        
        # A set to store hashes of paths, to avoid repetition
        path_hashes = set()

        # Initialize a queue with the start node and path as a list of one node - the start node.
        queue = [(start_node, [start_node_name], 1, [], 0)]  # Effect of path starting from start_node is 1 (multiplicative identity)
        paths = []  # This list will hold all the found paths.

        # Iterate over the queue, while it's not empty.
        while queue:
            node, path, effect, scores, outliers = queue.pop(0)  # Dequeue the first node in the queue.
            # If path length exceeds the cutoff, continue to the next iteration without processing current node.
            if len(path) > cut_off:
                continue

            # Generate a hash for the current path
            path_hash = hashlib.sha256(''.join(path).encode()).hexdigest()

            # If the path hash is in the set of path hashes, skip this iteration
            if path_hash in path_hashes:
                continue
            else:
                path_hashes.add(path_hash)

            # If the dequeued node is the end node, then a path is found. Add it and its effect to the paths list.
            if node == end_node:
                paths.append(path + [effect, self.calculate_IF_statistics(scores), outliers])

            # For each out edge of the node, if the destination node is not already in the current path,
            # enqueue the destination node and append it to the current path.
            # Also calculate the cumulative effect by multiplying the current effect and the edge effect.
            for edge in node.out_edges:
                if edge.to_node.name not in path:
                    new_scores = scores + [edge.score]
                    new_outliers = outliers + (1 if edge.score < 0.86 else 0)
                    queue.append((edge.to_node, path + [edge.to_node.name], effect * edge.effect, new_scores, new_outliers))

        return paths  # Return all found paths from the start node to the end node.
    
    def draw_network(self, layout='spring', k=None):
        """Draws the network graph
        Input: layout (string): The layout algorithm to use for the graph.
               k (float): Optimal distance between nodes for spring layout.
        Output: None. Draws the network graph using the specified layout algorithm."""
        fig, ax = plt.subplots()  # Explicitly create a figure and axis
        G = nx.DiGraph()
        
        # Add nodes and edges to the graph
        for node in self.nodes.values():
            G.add_node(node.name)
            for edge in node.out_edges:
                G.add_edge(edge.from_node.name, edge.to_node.name)
                
        # Decide the layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=k)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kaway':
            pos = nx.kamada_kaway_layout(G)
        else:
            print("Invalid layout type. Using spring layout.")
            pos = nx.spring_layout(G, k=k)

        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_cmap=plt.cm.Blues, font_size=10, ax=ax)
        plt.show()

    def save_network_to_txt(network, file_name="network_data.txt", combine = False, edge_only = False):
        """Save the network to a text file in the format:
        input: network (Network): The network to save.
               file_name (string): The name of the file to save the network to.
               combine (bool): If True, combine the nodes and edges in the same file.
               edge_only (bool): If True, only save the edges.
        output: None. Saves the network to a text file in the specified format."""
        # Open the file in write mode
        with open(file_name, "w") as file:
            if combine == True:
                edge_list = []
                
                if not edge_only:
                    # First, write all nodes
                    file.write("Nodes:\n")
                    for node_name in network.nodes:
                        file.write(node_name + "\n")
                
                # Then, write all edges
                file.write("\nEdges:\n")
                for node in network.nodes.values():
                    for edge in node.out_edges:
                        # Write edge details in the format:
                        # from_node -> to_node : effect, score, attribute
                        if ({edge.from_node.name} , {edge.to_node.name}) not in edge_list:
                            edge_list.append(({edge.from_node.name} , {edge.to_node.name}))
                            file.write(f"{edge.from_node.name} -> {edge.to_node.name} : {edge.effect}, {edge.score}, {edge.attribute}\n")
                        
            else:
                if not edge_only:
                    # First, write all nodes
                    file.write("Nodes:\n")
                    for node_name in network.nodes:
                        file.write(node_name + "\n")
                
                # Then, write all edges
                file.write("\nEdges:\n")
                for node in network.nodes.values():
                    for edge in node.out_edges:
                        # Write edge details in the format:
                        # from_node -> to_node : effect, score, attribute
                        file.write(f"{edge.from_node.name} -> {edge.to_node.name} : {edge.effect}, {edge.score}, {edge.attribute}\n")
        
    def calculate_IF_statistics(self, scores):
        """Calculate the minimum, 1st quartile, median, 3rd quartile, and maximum of a list of scores
        Input: scores (list): A list of scores to calculate statistics for.
        Output: A tuple containing the minimum, 1st quartile, median, 3rd quartile, and maximum of the scores."""
        if not scores:
            return (0, 0, 0, 0, 0)
        scores.sort()
        min_score = round(scores[0], 2)
        max_score = round(scores[-1], 2)
        quartile_1 = round(scores[len(scores) // 4], 2)
        median = round(scores[len(scores) // 2], 2)
        quartile_3 = round(scores[len(scores) * 3 // 4], 2)
        return (min_score, quartile_1, median, quartile_3, max_score)
        
    def FET_p_score_calculate_abcd(self, u, d):
        """
            Calculate the a, b, c, and d values for Fisher's Exact Test
            input: u, d are lists of up-regulated and down-regulated genes
            output: a, b, c, d values for Fisher's Exact Test
        """
        node_results = {}
        missing_gene_nodes = []
        modes = ['U', "D"]
        node_list = u + d
        gene_nodes = [self.nodes[node_name] for node_name in node_list if node_name in self.nodes]
        up_genes = [self.nodes[node_name] for node_name in u if node_name in self.nodes]
        down_genes = [self.nodes[node_name] for node_name in d if node_name in self.nodes]
        missing_gene_nodes = [node_name for node_name in node_list if node_name not in self.nodes]

        Upstream_nodes = set()
        for node in self.nodes.values():
            for edge in node.out_edges:
                if edge.attribute == "T" and edge.to_node in gene_nodes:
                    Upstream_nodes.add(node)
                    break
        
        total_T_edge_count = 0
        for node in Upstream_nodes:
            total_T_edge_count += sum(1 for edge in node.out_edges if edge.attribute == "T")
            
        Dataset_size = int(len(gene_nodes))
        
        for Upstream_node in Upstream_nodes:
            for mode in modes:
                # Initializing counters
                T_edge_regulate_list = 0

                total_T_edge = sum(1 for edge in Upstream_node.out_edges if edge.attribute == "T")
                
                for edge in Upstream_node.out_edges:
                    if edge.attribute == "T" and edge.to_node in gene_nodes:
                        # Check if in 'U' mode and the effect is either +1 (activate) or 3 (dual effector)
                        if mode == "U" and edge.effect != -1 and edge.to_node in up_genes: 
                            T_edge_regulate_list += 1
                        elif mode == "U" and edge.effect != 1 and edge.to_node in down_genes:
                            T_edge_regulate_list += 1
                        elif mode == "D" and edge.effect != 1 and edge.to_node in up_genes:
                            T_edge_regulate_list += 1
                        elif mode == "D" and edge.effect != -1 and edge.to_node in down_genes:
                            T_edge_regulate_list += 1
                
                node_results[(Upstream_node.name, mode)] = (
                    T_edge_regulate_list,
                    int(Dataset_size - T_edge_regulate_list),
                    int(total_T_edge - T_edge_regulate_list),
                    total_T_edge_count - (T_edge_regulate_list + 
                                          int(Dataset_size - T_edge_regulate_list) + 
                                          int(total_T_edge - T_edge_regulate_list))
                )  # (a, b, c, d = n - a - b - c)

        print("original_list_length:", len(node_list), "missing_node_count:", len(missing_gene_nodes))
        return node_results
    
    #################################################################### 
    #               #    Regulated    #   Not Regulated   #   Total   #
    #   In Dateset  #       a         #         b         #   a + b   #
    #     Not IN    #       c         #         d         #   c + d   #
    #     Total     #     a + c       #       b + d       #     n     #

    # a for gene included in datasets and are regulated
    # b for gene in datasets but not regulated
    # c for gene not in datasets but regulated
    # d are the rest genes, or total T edges minus (a+b+c)
    #################################################################### 

    ####################################################################   


def modified_fisher_exact(a, b, c, d,cut_off = 0.1):
    """calculate p value for Fisher's Exact Test
    input: a, b, c, d values for Fisher's Exact Test
    output: p value for Fisher's Exact Test"""
    n = a + b + c + d
    p_value = 0

    # Calculate initial parts of the numerator and denominator outside of the loop
    numerator_base = math.factorial(a + b) * math.factorial(b + d) * math.factorial(a + c) * math.factorial(c + d)
    denominator_base = math.factorial(n)
    
    for k in range(min(b, c) + 1):
        if b - k > 0: # avoid 
            denominator = denominator_base * math.factorial(a + k) * math.factorial(b - k) * math.factorial(c - k) * math.factorial(d + k)
        else:
            denominator = denominator_base * math.factorial(a + k) * 1 * math.factorial(c - k) * math.factorial(d + k)

        p_value += numerator_base / denominator
    
    if p_value <= cut_off: #th
        return p_value
    else:
        return str("Error, p_value >"+ str(cut_off) + "== " + str(p_value))

#################################################################### 
#               #    Regulated    #   Not Regulated   #   Total   #
#   In Dateset  #       a         #         b         #   a + b   #
#     Not IN    #       c         #         d         #   c + d   #
#     Total     #     a + c       #       b + d       #     n     #

# a for gene included in datasets and are regulated
# b for gene in datasets but not regulated
# c for gene not in datasets but regulated
# d are the rest genes, or total T edges minus (a+b+c)

def find_split(n=1, key='a', txt='ana', s=0, e=00):
    """Split a text and return a specific part of the split
    input: n (int): The index of the split to return.
           key (string): The string to split on.
           txt (string): The text to split.
           s (int): The starting index of the part to return.
           e (int): The ending index of the part to return.
    output: The n-th part of the split text."""
    key = str(key)
    txt = str(txt)
    if e == 00:
        e =len(txt)
    if n < 1:
        if txt.split(key)[0] != txt:
            return txt.split(key)[n][s:e].strip()
        else:
            return 0
    else:
        try:
            txt.split(key)[n]
            return txt.split(key)[n][s:e].strip()
        except:
            return 0
    
