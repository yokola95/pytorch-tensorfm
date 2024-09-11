import unittest
import torch

from src.torchfm.model.tensorfm import TensorFactorizationMachineModel

class TestTensorFactorizationMachineModel(unittest.TestCase):

    def test_interaction_matrix_calc(self):
        embed_dim = 4
        model = TensorFactorizationMachineModel([5,5,5,5], embed_dim, [2,3], [2,2])
        with torch.no_grad():
            x = torch.tensor([[[3,7,11,16], [3, 8, 12, 17]]])
            res = model(x)
            print(res)

            # diffTensor = torch.flatten(res1) - torch.flatten(res2)
            # diffTensorNorm = torch.linalg.vector_norm(diffTensor, ord=2)
            # assert diffTensorNorm < 1e-9

if __name__ == '__main__':
    unittest.main(verbosity=2)