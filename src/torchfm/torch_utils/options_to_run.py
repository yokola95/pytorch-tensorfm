import numpy as np
import itertools
from src.torchfm.torch_utils.constants import lowrank_fwfm, fwfm, pruned_fwfm, fm, logloss, auc

models_to_check_w_ranks = [lowrank_fwfm, pruned_fwfm]
models_to_check_w_no_ranks = [fm, fwfm]
models_to_check = models_to_check_w_ranks + models_to_check_w_no_ranks
metrics_to_optimize = [logloss]   # [logloss, auc]   # Does not matter in grid search
ranks_to_check = [1, 2, 3, 4, 5]
emb_sizes = [8, 16]
learning_rate = np.logspace(-4, -1, num=15).tolist()
reg_coef_vectors = [0.0, 0.001]
reg_coef_biases = [0.0, 0.001]
opt_name = ["adagrad"]
batch_size = [256]
options_for_studies_ranked = list(itertools.product(models_to_check_w_ranks,metrics_to_optimize,ranks_to_check, emb_sizes, learning_rate, opt_name, batch_size, reg_coef_vectors, reg_coef_biases))

options_for_studies_not_ranked = list(itertools.product(models_to_check_w_no_ranks,metrics_to_optimize, [0], emb_sizes, learning_rate, opt_name, batch_size, reg_coef_vectors, reg_coef_biases))

all_options_for_studies = options_for_studies_ranked + options_for_studies_not_ranked


class Option2Run:
    m_to_check = None
    met_to_opt = None
    rank = None
    emb_size = None
    lr = None
    opt_name = None
    batch_size = None
    return_l2 = None
    reg_coef_vectors = None
    reg_coef_biases = None
    part_id = None

    def __init__(self, m_to_check, met_to_opt, rank, emb_size, lr, opt_name, batch_size, reg_coef_vectors, reg_coef_biases, part_id):
        self.m_to_check = m_to_check
        self.met_to_opt = met_to_opt
        self.rank = rank
        self.emb_size = emb_size
        self.lr = lr
        self.opt_name = opt_name
        self.batch_size = batch_size
        self.reg_coef_vectors = reg_coef_vectors
        self.reg_coef_biases = reg_coef_biases
        self.part_id = part_id
        self.return_l2 = float(self.reg_coef_vectors) != 0.0 or float(self.reg_coef_biases) != 0.0

    def to_csv(self):
        res = ",".join([str(i) for i in [self.m_to_check, self.met_to_opt, self.rank, self.emb_size, self.lr, self.opt_name, self.batch_size, self.return_l2, self.reg_coef_vectors, self.reg_coef_biases, self.part_id]])
        return res


def tuple_to_option2run(tpl):
    return Option2Run(m_to_check=tpl[0], met_to_opt=tpl[1], rank=tpl[2], emb_size=tpl[3], lr=tpl[4], opt_name=tpl[5],
                      batch_size=tpl[6], return_l2=(float(tpl[8]) == 0.0 and float(tpl[9]) == 0.0), reg_coef_vectors=tpl[8], reg_coef_biases=tpl[9])
