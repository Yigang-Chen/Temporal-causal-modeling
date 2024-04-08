import os
import unittest
from Pathway_Intepreter import Network 

class TestNetworkRemoveDegreeOneNodes(unittest.TestCase):
    def setUp(self):
        #Set up the network structure for testing.#
        self.network = Network()
        # Construct the initial network: 
        # T -> A -> B -> C
        #      |    |
        #     \ /  \ /
        #      D1  D2
        edges = [
            ("T", "A", 1, 0.5, "P"),
            ("A", "B", 1, 0.5, "P"),
            ("B", "C", 1, 0.5, "P"),
            ("A", "D1", 1, 0.5, "T"),
            ("B", "D2", 1, 0.5, "T")
        ]
        for from_node, to_node, effect, score, attribute in edges:
            self.network.add_edge(from_node, to_node, effect, score, attribute)

    def test_network_structure_before_removal(self):
        #Test the network structure before removing degree-one nodes.#
        # Assert the existence of all nodes
        for node in ["T", "A", "B", "C", "D1", "D2"]:
            self.assertIn(node, self.network.nodes, f"Node {node} should exist in the network.")
        # Assert the specific edge existences
        self.assertTrue(self.network.has_edge("A", "B"), "Edge A->B should exist.")
        self.assertTrue(self.network.has_edge("B", "C"), "Edge B->C should exist.")

    def test_has_edge(self):
        #Test if the network correctly identifies existing edges.#
        self.assertTrue(self.network.has_edge("A", "B"), "The network should have an edge from A to B.")
        self.assertFalse(self.network.has_edge("C", "T"), "The network should not have an edge from C to T.")

    def test_remove_degree_one_nodes(self):
        #Test the network after removing degree-one nodes.#
        #network after remove degree one node
        # T -> A -> B 
        #      |    |
        #     \ /  \ /
        #      D1  D2
        self.network.remove_degree_one_nodes(start_nodes = ["T"], deg_up=["D1"], deg_down=["D2"])  # Adjust parameters as necessary
        # Assert the removal of C
        self.assertNotIn("C", self.network.nodes, "Node C should have been removed as a degree-one node.")
        # Assert the structure after removal
        for node in ["T", "A", "B", "D1", "D2"]:
            self.assertIn(node, self.network.nodes, f"Node {node} should still exist in the network.")
        self.assertFalse(self.network.has_edge("B", "C"), "Edge B->C should not exist after removal.")
    
    def test_intersect_with(self):
        #Test intersection of two networks.#
        other_network = Network()
        edges = [
            ("A", "B", 1, 0.5, "P"),  # Common edge
            ("T", "A", 1, 0.5, "P"),  # Common edge
            ("X", "Y", 1, 0.5, "P")   # Unique edge
        ]
        for from_node, to_node, effect, score, attribute in edges:
            other_network.add_edge(from_node, to_node, effect, score, attribute)

        intersection = self.network.intersect_with(other_network)
        self.assertTrue(intersection.has_edge("A", "B"), "Intersected network should have common edge A->B.")
        self.assertFalse(intersection.has_edge("X", "Y"), "Intersected network should not have unique edge X->Y.")

    def test_bfs_all_paths(self):
        #Test finding all paths from start to end within cutoff.#
        paths = self.network.bfs_all_paths("T", "D1", 3)  # Cutoff 3 ensures T->A->B->C is included
        expected_paths = [['T', 'A', 'D1', 1, (0.5, 0.5, 0.5, 0.5, 0.5), 2]]
        self.assertEqual(paths, expected_paths, "Should find exactly one path from T to C.") 
    
#edge cases
class TestNetworkEdgeCases(unittest.TestCase):
    def setUp(self):
        #Set up the network structure for testing.#
        self.network = Network()
        # Simple network setup
        edges = [
            ("A", "B", 1, 0.5, "P"),
            ("B", "C", 1, 0.5, "P"),
        ]
        for from_node, to_node, effect, score, attribute in edges:
            self.network.add_edge(from_node, to_node, effect, score, attribute)

    def test_remove_nonexistent_node(self):
        #Attempt to remove a node that doesn't exist.#
        initial_node_count = len(self.network.nodes)
        self.network.remove_node("X")  # Non-existent node
        self.assertEqual(len(self.network.nodes), initial_node_count, "Node count should not change.")

    def test_remove_nonexistent_edge(self):
        #Attempt to remove an edge that doesn't exist.#
        self.network.remove_edge("A", "C")  # Non-existent edge
        # Simply ensure no error is thrown, as the specific implementation of remove_edge may vary

    def test_network_with_isolated_node(self):
        #Test the network's behavior with an isolated node.#
        self.network.add_node("Isolated")
        self.assertIn("Isolated", self.network.nodes, "Isolated node should exist in the network.")
        self.assertFalse(self.network.nodes["Isolated"].in_edges, "Isolated node should have no in-edges.")
        self.assertFalse(self.network.nodes["Isolated"].out_edges, "Isolated node should have no out-edges.")

    def test_intersect_with_empty_network(self):
        #Intersect with an empty network.#
        empty_network = Network()
        intersection = self.network.intersect_with(empty_network)
        self.assertEqual(len(intersection.nodes), 0, "Intersection with an empty network should be empty.")

    def test_bfs_path_to_same_node(self):
        #Test BFS where start and end node are the same.#
        paths = self.network.bfs_all_paths("A", "A", 3)
        self.assertEqual(len(paths), 1, "Should find a single path from a node to itself.")
        self.assertEqual(paths[0][0], "A", "The path should start and end with the same node.")

    def test_bfs_no_path_exists(self):
        #Test BFS where no path exists.#
        self.network.add_node("Isolated")
        paths = self.network.bfs_all_paths("A", "Isolated", 3)
        self.assertEqual(len(paths), 0, "Should find no path between unconnected nodes.")

if __name__ == '__main__':
    unittest.main()
