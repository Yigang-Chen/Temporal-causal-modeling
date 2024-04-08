import unittest
from Pathway_Intepreter import Network 
from Pathway_Intepreter import modified_fisher_exact
class TestNetworkAnalysis(unittest.TestCase):

    def setUp(self):
        # Setup your network here
        self.network = Network()
        # Construct the initial network: 
        # T -> A  ->  B     C -> E
        #      |  \   |   / |
        #     \ /  \ \ / / \ /
        #     G1     D1     D2
        #G1 is not DEG

        self.network.add_node("E")
        edges = [
            ("T", "A", 1, 0.5, "P"),
            ("A", "B", 1, 0.5, "P"),
            ("B", "C", 1, 0.5, "P"),
            ("A", "G1", 1, 0.5, "T"),
            ("A", "D1", -1, 0.5, "T"),
            ("B", "D1", -1, 0.5, "T"),
            ("C", "D1", -1, 0.5, "T"),
            ("C", "D2", -1, 0.5, "T"),
            ("C", "E", -1, 0.5, "P"),
        ]
        for from_node, to_node, effect, score, attribute in edges:
            self.network.add_edge(from_node, to_node, effect, score, attribute)


    def test_URA_P_score_and_fisher_exact(self):
        u = ["D2"]  # Up-regulated genes
        d = ["D1"]  # Down-regulated genes
        
        abcd_results = self.network.URA_P_score_calculate_abcd(u, d)

        a, b, c, d = abcd_results[("A", "U")]
        ###################################################################
        #               #    Regulated    #   Not Regulated   #   Total   #
        #   In Dateset  #       a         #         b         #   a + b   #
        #     Not IN    #       c         #         d         #   c + d   #
        #     Total     #     a + c       #       b + d       #     n     #
        # a for gene included in datasets and are regulated (D1, neg * neg = Up)
        # b for gene in datasets but not regulated (D2)
        # c for gene not in datasets but regulated (G1)
        # d are the rest genes, or total T edges minus (a+b+c) (5 - 3)
        ################################################################### 
        
        # Verify a, b, c, d values are as expected
        self.assertEqual(a, 1, "Incorrect value of a")
        self.assertEqual(b, 1, "Incorrect value of b")
        self.assertEqual(c, 1, "Incorrect value of c")  
        self.assertEqual(d, 2, "Incorrect value of d")  

        # Calculate p-value using modified_fisher_exact
        p_value = modified_fisher_exact(a, b, c, d, cut_off=0.1)
        # Check if p_value is not an error message (indicative of a p-value <= cutoff)
        self.assertNotEqual("Error", p_value, "P-value calculation error")

        a, b, c, d = abcd_results[("A", "D")]
        #################################################################### 
        #               #    Regulated    #   Not Regulated   #   Total   #
        #   In Dateset  #       a         #         b         #   a + b   #
        #     Not IN    #       c         #         d         #   c + d   #
        #     Total     #     a + c       #       b + d       #     n     #

        # a for gene included in datasets and are regulated
        # b for gene in datasets but not regulated (D1, D2)
        # c for gene not in datasets but regulated (G1, D1) (The exact defination is given by formula from "Causal analysis approaches in Ingenuity Pathway Analysis")
        # d are the rest genes, or total T edges minus (a+b+c) (5 - 4)
        # There is ok  to have overlaped nodes, because the interpretation is just for understand. The actual value is defined by logic.
        #################################################################### 
        
        # Verify a, b, c, d values are as expected
        self.assertEqual(a, 0, "Incorrect value of a")
        self.assertEqual(b, 2, "Incorrect value of b")
        self.assertEqual(c, 2, "Incorrect value of c")  
        self.assertEqual(d, 1, "Incorrect value of d")  

        # Calculate p-value using modified_fisher_exact
        p_value = modified_fisher_exact(a, b, c, d, cut_off=0.1)
        # Check if p_value is not an error message (indicative of a p-value <= cutoff)
        self.assertNotEqual("Error", p_value, "P-value calculation error")

    #edge cases
    def test_network_with_edgeless_node(self):
        #Test URA_P_score with a node that has no edges.#
        u = ["EdgelessNode"]
        d = []
        results = self.network.URA_P_score_calculate_abcd(u, d)
        self.assertEqual(len(results), 0, "Should handle nodes without edges gracefully.")

    def test_empty_ud_lists(self):
        #Test URA_P_score with empty u and d lists.#
        u = []
        d = []
        results = self.network.URA_P_score_calculate_abcd(u, d)
        self.assertEqual(len(results), 0, "Should return empty results for empty u and d lists.")

    def test_modified_fisher_exact_with_zeros(self):
        #Test modified_fisher_exact function with zero values.#
        p_value = modified_fisher_exact(0, 0, 0, 0, 0.1)
        self.assertTrue(isinstance(p_value, str) and "Error" in p_value, "Should return an error for zero values.")

    def test_modified_fisher_exact_with_skewed_values(self):
        #Test modified_fisher_exact with skewed values.#
        a, b, c, d = 1, 10000, 1, 10000
        p_value = modified_fisher_exact(a, b, c, d, 0.1)
        self.assertTrue(p_value != "Error", "Should correctly calculate p-value for skewed values.")     

# Run the tests
if __name__ == '__main__':
    unittest.main()
