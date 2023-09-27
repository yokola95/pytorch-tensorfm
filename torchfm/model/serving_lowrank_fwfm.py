import time
import torch
import torch.nn as nn
from torchfm.model.fwfm import BaseFieldWeightedFactorizationMachineModel, PrunedFieldWeightedFactorizationMachineModel, \
    FieldWeightedFactorizationMachineModel
from torchfm.model.low_rank_fwfm import LowRankFieldWeightedFactorizationMachineModel


class ServingLowRankFwFM(BaseFieldWeightedFactorizationMachineModel):
    def __init__(self, num_features, embed_dim, num_fields, c, num_item_fields, is_multivalued=False):
        super(ServingLowRankFwFM, self).__init__(num_features, embed_dim, num_fields, is_multivalued)

        self.num_item_fields = num_item_fields
        self.diag_e = nn.Parameter(torch.empty(c))  # (c)
        self.U_c = nn.Parameter(torch.empty(c, num_fields - num_item_fields))  # (c, num_fields)
        self.U_i = nn.Parameter(torch.empty(c, num_item_fields))

        with torch.no_grad():
            nn.init.trunc_normal_(self.diag_e, std=0.01)
            nn.init.trunc_normal_(self.U_c, std=0.01)
            nn.init.trunc_normal_(self.U_i, std=0.01)

    def _calc_diag_d(self, U):       # the efficient way
        tmp = U.t() * self.diag_e
        return -(U * tmp.t()).sum(0)  # (num_fields)

    def calc_factorization_interactions(self, embeddings_c, embeddings_i):  # emb = (batch_size, num_fields, embedding_dim)
        """
        Calculates low rank factorization interactions
        """
        P_c = torch.matmul(self.U_c, embeddings_c)                                     # (batch_size, c, embedding_dim)
        P_i = torch.matmul(self.U_i, embeddings_i)
        P = P_c + P_i
        diag_d_c = self._calc_diag_d(self.U_c)                                                   # (num_fields)
        diag_d_i = self._calc_diag_d(self.U_i)

        term11 = (torch.square(embeddings_c).sum(-1) * diag_d_c).sum(1)                            # (batch_size) <- (batch_size, num_fields) * (num_fields).sum(1)
        term12 = (torch.square(embeddings_i).sum(-1) * diag_d_i).sum(1)

        term2 = (torch.square(P).sum(-1) * self.diag_e).sum(1)                         # (batch_size) <- (batch_size, c) * (c).sum(1)
        return (term11 + term12 + term2) / 2                                                     # (batch_size, 1)

    def calc_linear_term(self, x_c, x_i):
        # Biases (field weights)
        biases_sum1 = self.bias(x_c).squeeze().sum(-1)  # (batch_size1, 1)
        biases_sum2 = self.bias(x_i).squeeze().sum(-1)  # (batch_size2, 1)

        return self.w0 + biases_sum1 + biases_sum2  # (batch_size, 1)

    def forward(self, x_c, x_i):
        # Embedding layer
        embeddings_c = self.embeddings(x_c)      # (batch_size, num_fields, embedding_dim)
        embeddings_i = self.embeddings(x_i)

        # linear term = global bias and biases per feature (field weights)
        lin_term = self.calc_linear_term(x_c, x_i)  # (batch_size, 1)

        factorization_interactions = self.calc_factorization_interactions(embeddings_c, embeddings_i)  # (batch_size, 1)

        # Combine field interactions and factorization interactions
        output = lin_term + factorization_interactions
        return output  # (batch_size, 1)


def do_runtime_experiment():
    num_items_in_auction = 1000
    num_epochs = 1000  # num loop iterations
    num_features = 10000
    embed_dim = 8
    num_fields = 63
    c = 3
    num_item_fields = 39
    num_ctx_fields = num_fields - num_item_fields

    low_ind_bound = 1
    high_ind_bound = num_features - 1


    serving_lr_model = ServingLowRankFwFM(num_features, embed_dim, num_fields, c, num_item_fields)
    serving_lr_model.eval()
    low_rank_model = LowRankFieldWeightedFactorizationMachineModel(num_features, embed_dim, num_fields, c)
    low_rank_model.eval()

    topk = c * (num_fields + 1)
    pruned_model = PrunedFieldWeightedFactorizationMachineModel(num_features, embed_dim, num_fields, topk=topk)
    pruned_model.eval()

    fwfm_model = FieldWeightedFactorizationMachineModel(num_features, embed_dim, num_fields)
    fwfm_model.eval()

    def run_model(f_model):
        start = time.time()
        if f_model == "serving":

            for epoch in range(num_epochs):
                user_ctx = torch.randint(low=low_ind_bound, high=high_ind_bound, size=(1, num_ctx_fields), dtype=torch.long)
                item = torch.randint(low=low_ind_bound, high=high_ind_bound, size=(num_items_in_auction, num_item_fields), dtype=torch.long)
                res = serving_lr_model(user_ctx, item)

        elif f_model == "low_rank":
            for epoch in range(num_epochs):
                user_item = torch.randint(low=low_ind_bound, high=high_ind_bound, size=(num_items_in_auction, num_fields), dtype=torch.long)
                res = low_rank_model(user_item)

        elif f_model == "pruned":
            for epoch in range(num_epochs):
                user_item = torch.randint(low=low_ind_bound, high=high_ind_bound, size=(num_items_in_auction, num_fields), dtype=torch.long)
                res = pruned_model(user_item)

        else:  # fwfm
            for epoch in range(num_epochs):
                user_item = torch.randint(low=low_ind_bound, high=high_ind_bound, size=(num_items_in_auction, num_fields), dtype=torch.long)
                res = fwfm_model(user_item)

        end = time.time()
        print(f_model, end - start)

    with torch.no_grad():
        for f_model in ["serving", "low_rank", "pruned", "fwfm"]:
            run_model(f_model)
