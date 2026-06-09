import unittest
import torch
from src.model import CrossAttnDeepHybridCPI, MGNN, BidirectionalCrossAttention


class TestMGNN(unittest.TestCase):
    def setUp(self):
        self.model = MGNN(num_input_features=87, out_dim=640)
        self.batch_size = 4
        self.num_nodes = 100
        self.x = torch.randn(self.num_nodes, 87)
        self.edge_index = torch.randint(0, self.num_nodes, (2, 200))
        self.batch = torch.tensor([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)

    def test_forward(self):
        node_features, graph_features = self.model(self.x, self.edge_index, self.batch)
        self.assertEqual(node_features.shape[1], 456)
        self.assertEqual(graph_features.shape, (self.batch_size, 640))


class TestBidirectionalCrossAttention(unittest.TestCase):
    def setUp(self):
        self.model = BidirectionalCrossAttention(embed_dim=640, num_heads=8)
        self.batch_size = 4
        self.compound_features = torch.randn(self.batch_size, 50, 640)
        self.protein_features = torch.randn(self.batch_size, 100, 640)

    def test_forward(self):
        updated_compound, updated_protein, attn_c2p, attn_p2c = self.model(
            self.compound_features, self.protein_features
        )
        self.assertEqual(updated_compound.shape, self.compound_features.shape)
        self.assertEqual(updated_protein.shape, self.protein_features.shape)
        self.assertEqual(attn_c2p.shape, (self.batch_size, 8, 50, 100))
        self.assertEqual(attn_p2c.shape, (self.batch_size, 8, 100, 50))


class TestCrossAttnDeepHybridCPI(unittest.TestCase):
    def setUp(self):
        self.model = CrossAttnDeepHybridCPI()
        self.batch_size = 4
        self.num_nodes = 100
        self.x = torch.randn(self.num_nodes, 87)
        self.edge_index = torch.randint(0, self.num_nodes, (2, 200))
        self.batch = torch.tensor([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
        self.protein_embeddings = torch.randn(self.batch_size, 640)

    def test_forward(self):
        logits = self.model(self.x, self.edge_index, self.batch, self.protein_embeddings)
        self.assertEqual(logits.shape, (self.batch_size, 2))

    def test_forward_with_attention(self):
        logits, attention = self.model(
            self.x, self.edge_index, self.batch, self.protein_embeddings,
            return_attention=True
        )
        self.assertEqual(logits.shape, (self.batch_size, 2))
        self.assertEqual(len(attention), 2)


if __name__ == '__main__':
    unittest.main()
