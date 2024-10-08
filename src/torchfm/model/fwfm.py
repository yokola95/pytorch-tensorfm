from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import numpy as np
from torch.nn.modules.module import T

from src.torchfm.model.embedding_factory import EmbeddingFactory
from src.torchfm.torch_utils.constants import sparseGrads


class BaseFieldWeightedFactorizationMachineModel(nn.Module):

    def __init__(self, num_features, embed_dim, num_fields, is_multival=False):
        super(BaseFieldWeightedFactorizationMachineModel, self).__init__()

        # num_features -- number of different values over all samples, num_fields -- number of columns
        self.num_features = np.max(num_features) + 1  # number of entries for embedding = (max possible ind.) +1 to since indexing starting from 0
        self.embedding_dim = embed_dim
        self.num_fields = num_fields  # length of X

        self.w0 = nn.Parameter(torch.zeros(1))  # w0 global bias
        self.bias = EmbeddingFactory.get_embedding(self.num_features, 1, sparse=sparseGrads, is_multival=is_multival)  # biases w: for every field 1 dimension embedding (num_features, 1)

        self.embeddings = EmbeddingFactory.get_embedding(self.num_features, embed_dim, sparse=sparseGrads, is_multival=is_multival)  # embedding vectors V: (num_features, embedding_dim)

        with torch.no_grad():
            nn.init.trunc_normal_(self.bias.weight, std=0.01)
            nn.init.trunc_normal_(self.embeddings.weight, std=0.01)

    def calc_linear_term(self, x, return_l2=False):
        # Biases (field weights)
        biases_x, _ = self.bias(x)
        biases_sum = biases_x.squeeze().sum(-1)  # (batch_size, 1)
        score = self.w0 + biases_sum  # (batch_size, 1)
        if return_l2:
            lin_reg = biases_x.square().mean(0).sum()     # + torch.square(self.w0)   no need to add global bias to the regularization
            return score, lin_reg
        else:
            return score, 0.0

    def forward(self, x, return_l2=False):
        """
        :param x: Float tensor of size ``(batch_size, num_fields)``
        :param return_l2: flag showing whther to return l2 regularization term
        """
        # Embedding layer
        emb, _ = self.embeddings(x)  # (batch_size, num_fields, embedding_dim)

        # linear term = global bias and biases per feature (field weights)
        lin_term, lin_reg = self.calc_linear_term(x, return_l2)   # (batch_size, 1)

        factorization_interactions = self.calc_factorization_interactions(emb)  # (batch_size, 1)
        if return_l2:
            reg = self.get_l2_reg(emb)
        else:
            reg = 0.0

        # Combine field interactions and factorization interactions
        output = lin_term + factorization_interactions
        total_reg = [reg, lin_reg]

        return output, total_reg  # (batch_size, 1)

    @abstractmethod
    def calc_factorization_interactions(self, emb):
        pass

    def get_l2_reg(self, emb):
        reg = emb.square().mean(0).sum()
        return reg


class Symmetric(nn.Module):
    def forward(self, x):
        return x.triu(1) + x.triu(1).transpose(-1, -2)  # zero diagonal and symmetric - due to parametrization no need to remove diagonal in the code


class FieldWeightedFactorizationMachineModel(BaseFieldWeightedFactorizationMachineModel):
    #  __use_tensors_field_interact_calc = True

    def __init__(self, num_features, embed_dim, num_fields, is_multivalued=False):
        super(FieldWeightedFactorizationMachineModel, self).__init__(num_features, embed_dim, num_fields, is_multivalued)
        self.field_inter_weights = self._init_interaction_weights(num_fields)
        parametrize.register_parametrization(self, "field_inter_weights", Symmetric())

    def calc_factorization_interactions(self, emb):  # emb = (batch_size, num_fields, embedding_dim)
        return self._calc_factorization_interactions_tensors(emb)

    def _init_interaction_weights(self, num_fields):
        aux = torch.empty(num_fields, num_fields)
        with torch.no_grad():
            nn.init.trunc_normal_(aux, std=0.01)
        return nn.Parameter(aux)

    # Mostly for debugging (comparison for an equality of the output)
    def _calc_factorization_interactions_nested_loops(self, emb):
        factorization_interactions = 0.0
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                inner_prod = torch.sum(emb[..., i, :] * emb[..., j, :], dim=-1)
                factorization_interactions += self.field_inter_weights[i][j] * inner_prod

        return factorization_interactions

    def _calc_factorization_interactions_tensors(self, emb):  # emb = (batch_size, num_fields, embedding_dim)
        emb_mul_emb_T = torch.matmul(emb, torch.transpose(emb, -2, -1))
        inner_product = (emb_mul_emb_T * self.field_inter_weights).sum([-1, -2])  # inner_product = (batch_size, 1)   # due to parametrization, self.field_inter_weights is zero-diagonal and symmetric matrix of size (num_fields, num_fields)
        return inner_product / 2  # (batch_size, 1)

    def get_l2_reg(self, emb):
        base_reg = super(FieldWeightedFactorizationMachineModel, self).get_l2_reg(emb)
        iter_reg = self.field_inter_weights.square().mean() / 2
        return base_reg + iter_reg


class PrunedFieldWeightedFactorizationMachineModel(FieldWeightedFactorizationMachineModel):
    _use_topk = False
    _topk = -1
    _topk_vals = None
    _topk_rows = None
    _topk_columns = None

    _R_sparse = None

    def __init__(self, num_features, embed_dim, num_fields, topk, is_multivalued=False):
        super(PrunedFieldWeightedFactorizationMachineModel, self).__init__(num_features, embed_dim, num_fields, is_multivalued)
        self._topk = topk

    def do_pruning_to_sparse(self, original_tensor):
        new_tensor = torch.zeros(original_tensor.size())
        for i, j in zip(self._topk_rows, self._topk_columns):
            new_tensor[i, j] = original_tensor[i, j]
        self._R_sparse = new_tensor.to_sparse_csc()

    def set_top_entries_from_field_inter_weights(self):
        if self._topk <= 0:
            self._topk_vals = None
            self._topk_rows = None
            self._topk_columns = None
            return

        original_tensor = self.field_inter_weights.triu(1)  # 2D tensor
        input_tensor = torch.abs(original_tensor)  # 2D tensor
        flat_tensor = input_tensor.view(-1)
        topk_values, topk_indices_flat = torch.topk(flat_tensor, self._topk)  # from 1D tensor

        # Convert the flat indices back to 2D indices
        # topk_indices_2d = torch.tensor([divmod(idx.item(), input_tensor.shape[1]) for idx in topk_indices_flat])

        self._topk_rows = torch.div(topk_indices_flat, input_tensor.shape[1], rounding_mode='trunc').detach()
        self._topk_columns = topk_indices_flat % input_tensor.shape[1]

        # self.do_pruning_to_sparse(original_tensor)

        # The original values
        self._topk_vals = original_tensor[self._topk_rows, self._topk_columns].detach()

    def calc_factorization_interactions_debug(self, emb):
        if not self._use_topk:
            return super(PrunedFieldWeightedFactorizationMachineModel, self).calc_factorization_interactions(emb)

        if self._topk <= 0:
            return 0.0

        factorization_interactions = 0.0
        for val, i, j in zip(self._topk_vals, self._topk_rows, self._topk_columns):
            inner_prod = torch.sum(emb[..., i, :] * emb[..., j, :], dim=-1)
            factorization_interactions += val * inner_prod
        return factorization_interactions

    def calc_factorization_interactions(self, emb):
        if not self._use_topk:
            return super(PrunedFieldWeightedFactorizationMachineModel, self).calc_factorization_interactions(emb)

        if self._topk <= 0:
            return 0.0

        emb_i = torch.index_select(emb, -2, index=self._topk_rows)
        emb_j = torch.index_select(emb, -2, index=self._topk_columns)
        factorization_interactions = torch.sum(torch.sum(emb_i * emb_j, dim=-1) * self._topk_vals, dim=-1)
        return factorization_interactions

    def calc_factorization_interactions_sparse(self, emb):   # sparse representation
        if not self._use_topk:
            return super(PrunedFieldWeightedFactorizationMachineModel, self).calc_factorization_interactions(emb)

        if self._topk <= 0:
            return 0.0

        U = emb.transpose(-1, -2).clone(memory_format=torch.contiguous_format)
        UR = torch.matmul(U, self._R_sparse)

        factorization_interactions = 0.5 * (UR.transpose(-1, -2) * emb).sum([-1, -2])
        return factorization_interactions

    def train(self: T, mode: bool = True) -> T:
        super(PrunedFieldWeightedFactorizationMachineModel, self).train(mode)
        if not mode and not self._use_topk:
            self.set_top_entries_from_field_inter_weights()
        self._use_topk = not mode
