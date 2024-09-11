import torch
import torch.nn as nn
from src.torchfm.layer import FeaturesEmbedding, FeaturesLinear

class TensorFactorizationMachineModel(torch.nn.Module):
    # dim_int = [d_1,...,d_l] and rank_tensors = [r_1,...,r_l] are two lists
    # For an index 1 <= i <= l, we consider a d_i-order interaction of rank r_i
    def __init__(self, field_dims, embed_dim, dim_int, rank_tensors, is_multivalued=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, is_multival=is_multivalued)
        self.linear = FeaturesLinear(field_dims, is_multival=is_multivalued)
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        self.dim_int = dim_int
        self.l = len(dim_int)
        self.rank_tensors = rank_tensors
        
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(dim_int[i], rank_tensors[i], self.num_fields))  
            for i in range(self.l)
        ])

        with torch.no_grad():
            for i in range(self.l):
                nn.init.trunc_normal_(self.W[i], std=0.01)

    def forward(self, x): # A = (batch_size, num_fields, embedding_dim)
        emb,_ = self.embedding(x)
        ret,_ = self.linear(x)

        for i in range(self.l):
            ret = torch.add(ret, self.calc_cross(emb,i))
        return ret,0

    def calc_cross(self,A,idx):  # efficiently
            W_i = self.W[idx]  # W_i has shape (d, r, n)
            W_i = W_i.unsqueeze(0)  # Now W_i has shape (1, d, r, n)
            #print("W_i,size", W_i.size())
            #print("A,size",A.size() )
            dot_products = torch.einsum('bljk,bki->bijl',W_i,A) # (b, emb_dim, r, d)
            prod_over_d = torch.prod(dot_products,axis=3)
            sum_all = torch.sum(prod_over_d, axis=[1,2])
            #print("SUM ALL", sum_all)
            return sum_all.unsqueeze(1)


    def get_l2_reg(self, emb):
        base_reg = self.embedding.get_l2_reg(emb) + self.linear.get_l2_reg(emb),
        tensor_reg = nn.sum(self.W.square().mean())
        return base_reg + tensor_reg
