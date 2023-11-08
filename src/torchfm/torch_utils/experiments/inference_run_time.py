import time
import torch
from src.torchfm.model.fm import FactorizationMachineModel
from src.torchfm.model.fwfm import PrunedFieldWeightedFactorizationMachineModel, FieldWeightedFactorizationMachineModel
from src.torchfm.model.low_rank_fwfm import LowRankFieldWeightedFactorizationMachineModel
from src.torchfm.model.serving_lowrank_fwfm import ServingLowRankFwFM


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

    fm_model = FactorizationMachineModel(num_features, embed_dim)
    fm_model.eval()


    def run_standard_user_item(model):
        for epoch in range(num_epochs):
            user_item = torch.randint(low=low_ind_bound, high=high_ind_bound, size=(num_items_in_auction, num_fields),
                                      dtype=torch.long)
            res = model(user_item)

    def run_user_ctx_item(model):
        for epoch in range(num_epochs):
            user_ctx = torch.randint(low=low_ind_bound, high=high_ind_bound, size=(1, num_ctx_fields), dtype=torch.long)
            item = torch.randint(low=low_ind_bound, high=high_ind_bound, size=(num_items_in_auction, num_item_fields),
                                 dtype=torch.long)
            res = model(user_ctx, item)

    all_models = {"serving": (serving_lr_model, run_user_ctx_item),
                  "low_rank": (low_rank_model, run_standard_user_item),
                  "pruned": (pruned_model, run_standard_user_item),
                  "fwfm": (fwfm_model, run_standard_user_item),
                  "fm": (fm_model, run_standard_user_item)}

    with torch.no_grad():
        for model_nm in all_models:
            model = all_models[model_nm][0]
            run_model_fn = all_models[model_nm][1]

            start = time.time()
            run_model_fn(model)
            end = time.time()
            print(model_nm, end - start, model.get_time())


do_runtime_experiment()
