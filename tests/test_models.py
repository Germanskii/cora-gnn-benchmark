import copy
import unittest
import torch
from src.benchmark import GCN, GAT, GraphTransformer

class ModelsTest(unittest.TestCase):
    def test_shapes_and_checkpoint_independence(self):
        x = torch.randn(6, 8)
        edges = torch.tensor([[0,1,2,3,4,5,1,2],[1,2,3,4,5,0,0,1]])
        for cls in (GCN,GAT,GraphTransformer):
            with self.subTest(model=cls.__name__):
                model=cls(8,4,3).eval()
                output=model(x,edges)
                self.assertEqual(output.shape,(6,3))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.allclose(output.exp().sum(1),torch.ones(6),atol=1e-5))
                saved=copy.deepcopy(model.state_dict())
                key=next(iter(saved));before=saved[key].clone()
                with torch.no_grad(): next(model.parameters()).add_(1)
                self.assertTrue(torch.equal(saved[key],before))
