from src.main_functions.main import top_main_for_option_run


def run_save_single_model():
    opt_name = "adagrad"
    batch_size = 256
    learning_rate = 0.01578378074
    model_name = "lowrank_fwfm"
    device_ind = 0
    metric_to_optimize = "logloss"
    emb_size = 8
    rank_param = 5
    study = None
    trial = None
    reg_coef_vectors = 0.01
    reg_coef_biases = 0.01

    from src.torchfm.torch_utils.options_to_run import Option2Run
    option_to_run = Option2Run(model_name, metric_to_optimize, rank_param, emb_size, learning_rate, opt_name, batch_size, reg_coef_vectors, reg_coef_biases, 0)

    top_main_for_option_run(study, trial, device_ind, option_to_run)


def show_image(arr):
    # transform = T.ToPILImage()
    # img = transform(field_inter_weights)
    # img.show()

    import matplotlib.pyplot as plt
    import numpy as np

    #arr = np.ndarray((1,80,80,1))#This is your tensor
    #fig, ax = plt.subplots(figsize = (40,40))
    #ax.set_title('Interaction Weights of FwFM')
    #ax.set_xlabel('Rows')
    #ax.set_ylabel('Columns')

    arr_ = np.squeeze(arr)  # you can give axis attribute if you wanna squeeze in specific dimension
    plt.imshow(arr_)

    # Add colorbar
    min = np.min(arr_)
    max = np.max(arr_)
    middle = (max+min)/2
    cbar = plt.colorbar(ticks=[min, middle, max])
    #cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])

    plt.title("Interaction Weights of FwFM")
    plt.xlabel('Rows')
    plt.ylabel('Columns')
    plt.show()


#field_inter_weights = load_tensor("fwfm_0157_8")
#arr = field_inter_weights.detach().numpy()
#show_image(arr)

# run_save_single_model()
