import numpy as np 
from matplotlib import patches
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
'''
written by Elena Zoppellari
'''

def basic_selectivity(L_act, R_act, start_d, end_d):
    mean_across_trials_right_c = np.mean(R_act[start_d:end_d,:], axis=0)
    mean_across_trials_left_c = np.mean(L_act[start_d:end_d,:], axis=0)
    selectivity = (mean_across_trials_right_c - mean_across_trials_left_c) / (mean_across_trials_right_c + mean_across_trials_left_c)
    sel = []
    for i in range(R_act.shape[1]):
        if selectivity[i] > 0.1:
            sel.append("right")
        elif selectivity[i] < -0.1:
            sel.append("left")
        else:
            sel.append("none")
    return sel

def plotting(list, n, t_list_double_cut, r_p, r_p_random, J_matrix_p, J_matrix_p_random, n_iters, pvar, start_cue, start_cue_right_c, start_sample_right_c, start_delay_right_c):
    times_left_c = np.ones((201, 6000))[:, 1000:(start_cue+250)]*np.linspace(0, 6, 6000)[1000:(start_cue+250)] #4778
    times_right_c = np.ones((201, 6000))[:, 1000:(start_cue+250)]*np.linspace(0, 6, 6000)[1000:(start_cue+250)]
    time_ax_r = np.nanmean(times_right_c, axis=0) - np.nanmean(start_cue_right_c)
    for i in list:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        ax[0].set_title(f"Avg correct Lick-Right trials, {n_iters} iterations, pVar={pvar[0]}%")
        ax[1].set_title(f"Avg correct Lick-Left trials, {n_iters} iterations, pVar={pvar[1]}%")
        ax[0].plot(time_ax_r, r_p[i][0][:,n], label="trained net pred", color="purple")
        ax[0].plot(time_ax_r, r_p_random[i][0][:,n], label="random net pred", color="orange")
        ax[0].plot(time_ax_r, t_list_double_cut[0][n], label="true", color="blue")
        ax[1].plot(time_ax_r, r_p[i][1][:,n], label="pred r trained net", color="purple")
        ax[1].plot(time_ax_r, t_list_double_cut[1][n], label="true", color="red")
        ax[1].plot(time_ax_r, r_p_random[i][1][:,n], label="pred r random net", color="orange")
        for i in range(2):
            ax[i].axvline(np.nanmean(start_sample_right_c) - np.nanmean(start_cue_right_c), linestyle="--", alpha=0.5, label="signal onset")
            ax[i].axvspan(np.nanmean(start_delay_right_c) - np.nanmean(start_cue_right_c), 0, alpha=0.2, color='greenyellow', label="delay")
            ax[i].set_xlabel("Time to movement onset (s)")
            ax[i].set_ylabel("R(t)")
        ax[0].legend()
        ax[1].legend()
        plt.show()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
        vmin = min(J_matrix_p[i].min(), J_matrix_p_random[i].min())
        vmax = max(J_matrix_p[i].max(), J_matrix_p_random[i].max())
        v = max(vmax, (-vmin))
        im0 = ax[0].matshow(J_matrix_p[i], cmap="seismic", vmin=(-v), vmax=v)
        im1 = ax[1].matshow(J_matrix_p_random[i], cmap="seismic", vmin=(-v), vmax=v)
        ax[0].set_title("N=87, trained network")
        ax[1].set_title("N=87, random network")
        # Add a single colorbar that matches both plots
        fig.colorbar(im0, ax=ax[0], shrink=0.3)
        fig.colorbar(im1, ax=ax[1], shrink=0.3)
        plt.tight_layout()
        plt.show()

def generate_train_test(N, array, common):
    #idx_train = np.random.choice(N, N//2, replace=False)
    idx_train = common
    idx_test = np.setdiff1d(np.arange(N), idx_train)
    activity_train = array[:, idx_train, :]
    activity_test = array[:, idx_test, :]
    return activity_train, activity_test

# calculate norm factors
def norm_factor(cd_vector, array, shared_start_cue_idx):
    trials_proj = np.dot(cd_vector, array[:, :, shared_start_cue_idx])
    return np.nanmedian(trials_proj)

def calc_edges(proj, shared_start_cue_idx):
    edges = proj[:, shared_start_cue_idx]
    clean_edges = edges[~np.isnan(edges)]
    bin_edges = np.linspace(-2,2,20)
    bin_centers = [(bin_edges[i+1]-bin_edges[i])/2 + bin_edges[i] for i in range(bin_edges.shape[0]-1)]
    counts, _ = np.histogram(clean_edges, bins=bin_edges)
    return bin_centers, counts/clean_edges.shape[0]

def CD_analysis(psth_right_c, psth_left_c, plot, start_cue, start_sample, start_delay):
    # first step: divide train and test correct trials randomly
    # we need the contraint to have in the train set not nan values
    start_cue_idx_left_c = np.ones(201)*(start_cue - 1000)
    start_cue_left_c = np.ones(201)*np.linspace(0, 6, 6000)[start_cue]
    times_left_c = np.ones((201, 6000))[:,1000:(start_cue+250)]*np.linspace(0, 6, 6000)[1000:(start_cue+250)] 
    start_sample_right_c = np.ones(201)*np.linspace(0, 6, 6000)[start_sample]
    start_delay_right_c = np.ones(201)*np.linspace(0, 6, 6000)[start_delay]
    start_cue_right_c = np.ones(201)*np.linspace(0, 6, 6000)[start_cue]
    times_right_c = np.ones((201, 6000))[:,1000:(start_cue+250)]*np.linspace(0, 6, 6000)[1000:(start_cue+250)]
    n_train_R = psth_right_c.shape[1] // 2
    n_train_L = psth_left_c.shape[1] // 2
    train_R, test_R = generate_train_test(psth_right_c.shape[1], psth_right_c, np.arange(0,n_train_R))
    train_L, test_L = generate_train_test(psth_left_c.shape[1], psth_left_c, np.arange(0,n_train_L))
    
    # CD creation
    avg_R, avg_L = np.nanmean(train_R, axis=1), np.nanmean(train_L, axis=1)
    weight_vector = avg_R - avg_L
    shared_start_cue_idx = int(start_cue_idx_left_c[0])
    w_delay = np.nanmean(weight_vector[:, (shared_start_cue_idx-600):shared_start_cue_idx], axis=1)
    cd_vector = w_delay / np.linalg.norm(w_delay) 
    norm_right = norm_factor(cd_vector, train_R, shared_start_cue_idx)
    norm_left = norm_factor(cd_vector, train_L, shared_start_cue_idx)
    
    # projections 
    proj_R = (np.dot(cd_vector, np.transpose(test_R, axes=(1,0,2))) - norm_left) / ( norm_right - norm_left)
    proj_L = (np.dot(cd_vector, np.transpose(test_L, axes=(1,0,2))) - norm_left) / ( norm_right - norm_left)
    
    # compute edges
    x_R, edges_R = calc_edges(proj_R, shared_start_cue_idx)
    x_L, edges_L = calc_edges(proj_L, shared_start_cue_idx)

    # avg of projections between trials
    mean_test_r = np.nanmean(proj_R, axis=0)
    std_test_r = np.nanstd(proj_R, axis=0) / np.sqrt(np.sum(~np.isnan(proj_R).all(axis=1)))
    time_ax_r = np.nanmean(times_right_c, axis=0) - np.nanmean(start_cue_right_c)
    mean_test_l = np.nanmean(proj_L, axis=0)
    std_test_l = np.nanstd(proj_L, axis=0) / np.sqrt(np.sum(~np.isnan(proj_L).all(axis=1)))
    time_ax_l = np.nanmean(times_left_c, axis=0)  - np.nanmean(start_cue_left_c)

    if plot == True:
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        
        # plot avg projection R and L
        ax[0].plot(time_ax_r[10:-10], mean_test_r[10:-10], color='blue', alpha=0.5, label='Lick-right')
        ax[0].axhline(1, alpha=0.1, linestyle="--")
        ax[0].plot(time_ax_l[10:-10], mean_test_l[10:-10], color='red', alpha=0.5, label='Lick-left')
        ax[0].axhline(0, alpha=0.1, linestyle="--")
        ax[0].fill_between(time_ax_r[10:-10], 
             mean_test_r[10:-10] + std_test_r[10:-10], 
             mean_test_r[10:-10] - std_test_r[10:-10], 
             color="blue", alpha=0.3, label="Right STD")
        
        ax[0].fill_between(time_ax_l[10:-10], 
                         mean_test_l[10:-10] + std_test_l[10:-10], 
                         mean_test_l[10:-10] - std_test_l[10:-10], 
                         color="red", alpha=0.3, label="Left STD")
        ax[0].set_ylabel('Proj. to CD')
        ax[0].set_xlabel('Time to movement onset (s)')
        ax[0].axvline(np.nanmean(start_sample_right_c) - np.nanmean(start_cue_right_c), linestyle="--", alpha=0.5, label="signal onset")
        ax[0].axvspan(np.nanmean(start_delay_right_c) - np.nanmean(start_cue_right_c), 0, alpha=0.2, color='greenyellow', label="delay")
        ax[0].set_title("Average projection of trials in the session")
        ax[0].legend()
        
        ax[1].plot(x_R, edges_R, color="blue", alpha=0.7, label='Lick-right')
        ax[1].plot(x_L, edges_L, color="red", alpha=0.75, label='Lick-left')
        ax[1].axvline(1, linestyle="--", alpha=0.1)
        ax[1].axvline(0, linestyle="--", alpha=0.1)
        ax[1].set_xlabel('Proj. to CD')
        ax[1].set_ylabel('Fraction of trials')
        ax[1].set_title(f'Trials endpoint distribution')
        ax[1].legend()
        
    return cd_vector, norm_right, norm_left, time_ax_r, time_ax_l, shared_start_cue_idx