import unittest
import torch

from src.torchfm.model.fwfm import PrunedFieldWeightedFactorizationMachineModel


class TestPrunedFieldWeightedFactorizationMachineModel(unittest.TestCase):

    def test_interaction_matrix_calc(self):
        num_features = 10
        embed_dim = 4
        num_fields = 5
        model = PrunedFieldWeightedFactorizationMachineModel(num_features, embed_dim, num_fields, 2)
        model.eval()

        with torch.no_grad():
            x = torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])
            emb = model.embeddings(x)
            res1 = model.calc_factorization_interactions(emb)
            res2 = model.calc_factorization_interactions_debug(emb)
            assert torch.allclose(res1, res2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
