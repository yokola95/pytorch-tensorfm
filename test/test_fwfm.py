import unittest

import numpy as np
import torch

from torchfm.model.fwfm import FieldWeightedFactorizationMachineModel
from torchfm.model.low_rank_fwfm import LowRankFieldWeightedFactorizationMachineModel

class TestFieldWeightedFactorizationMachineModel(unittest.TestCase):

    def test_interaction_matrix_calc(self):
        num_features = 10
        embed_dim = 4
        num_fields = 5
        model = FieldWeightedFactorizationMachineModel(num_features, embed_dim, num_fields)
        with torch.no_grad():
            x = torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])
            emb = model.embeddings(x)
            res1 = model._calc_factorization_interactions_tensors(emb)
            res2 = model._calc_factorization_interactions_nested_loops(emb)
            diffTensor = torch.flatten(res1) - torch.flatten(res2)
            diffTensorNorm = torch.linalg.vector_norm(diffTensor, ord=2)

            assert diffTensorNorm < 1e-9

    def test_low_rank_interaction_matrix_calc(self):
        num_features = 10
        embed_dim = 4
        num_fields = 5
        c = 4
        model = LowRankFieldWeightedFactorizationMachineModel(num_features, embed_dim, num_fields, c)
        with torch.no_grad():
            x = torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])

            res1 = model._calc_diag_d()
            res2 = model._calc_diag_d_straighforward()
            diffTensor = torch.flatten(res1) - torch.flatten(res2)
            diffTensorNorm = torch.linalg.vector_norm(diffTensor, ord=2)

            assert diffTensorNorm < 1e-9

if __name__ == '__main__':
    unittest.main(verbosity=2)
