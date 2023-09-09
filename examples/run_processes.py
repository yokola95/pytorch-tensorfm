from multiprocessing import Process
from main_optuna import run_all_for_model

models_to_check = [('fwfm', 0), ('lowrank_fwfm', 1)]


if __name__ == '__main__':
    processes = [Process(target=run_all_for_model, args=(m[0], m[1]), daemon=True) for m in models_to_check]
    [p.start() for p in processes]
    print("started!!!!")
    [p.join() for p in processes]
    print("ended!!!!")
