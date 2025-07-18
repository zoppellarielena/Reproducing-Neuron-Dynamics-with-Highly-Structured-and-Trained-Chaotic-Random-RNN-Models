import numpy as np
from scipy.stats import ranksums
import matplotlib.pyplot as plt
''' written by Elena Zoppellari'''

def smooth(a, window_len):
    # a: NumPy 1-D array containing the data to be smoothed
    # window_len: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(window_len,dtype=int),'valid')/window_len    
    r = np.arange(1,window_len-1,2)
    start = np.cumsum(a[:window_len-1])[::2]/r
    stop = (np.cumsum(a[:-window_len:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def trial_matrix(n_unit, units):
    '''this function creates psth data and organize them into a numpy matrix'''
    unit = units[n_unit]
    
    # load trials indices
    trial_type = unit['Behavior']['Trial_types_of_response_vector'][0][0].flatten() # (n_trials, )
    trial_range = unit['Trial_info']['Trial_range_to_analyze'][0][0].flatten() # (2, )
    
    # Extract spikes from each trial
    spike_times = unit['SpikeTimes'].flatten()
    trial_idx_of_spikes = unit['Trial_idx_of_spike'].flatten()
    
    # psth times
    T_Axis_PSTH = unit['Meta_data']['parameters'][0][0]['T_Axis_PSTH'][0][0][0] # (6001,)

    #spike_collection = np.zeros((trial_type.shape[0], (T_Axis_PSTH.shape[0]-1)))
    bin_collection = np.zeros((trial_type.shape[0], (T_Axis_PSTH.shape[0]-1)))
    psth_collection = np.zeros((trial_type.shape[0], (T_Axis_PSTH.shape[0]-1)))
    
    # Process each trial: range (0, n_trials)
    for tr in range(0, trial_type.shape[0]):
        # extract the specific spikes for the trial
        trial_idx = tr
        spikes_tmp = spike_times[trial_idx_of_spikes == (tr+1)] #- cue_start[tr] 
        psth_tmp = np.histogram(spikes_tmp, bins=T_Axis_PSTH)#[0] * 1000 
        # calculate bin_centers
        bin_centers = [(psth_tmp[1][i] + (psth_tmp[1][i+1] - psth_tmp[1][i])/2) for i in range(psth_tmp[1].shape[0] - 1)]
        bin_collection[trial_idx,:] = bin_centers
        if (tr < (trial_range[0] - 1)) or (tr > trial_range[1]):
            psth_collection[trial_idx, :] = np.NaN
        else:
            psth_collection[trial_idx, :] = smooth(psth_tmp[0]*1000, 101) #* 1000
        
    SpikeWidth = unit['SpikeWidth'][0][0]
    
    return psth_collection, bin_collection, trial_range, SpikeWidth

def calc_idx(times, value):
    '''calcualte the corresponding index in the time axis for a specific time value'''
    indices = np.zeros(value.shape, dtype=int)
    for trial in range(value.shape[0]):
        indices[trial] = int(np.argmin(np.abs(times[trial] - value[trial])))
    return indices

def align_cue(start_cue_idx, all_times_array, psth_array, start_sample_idx, start_delay_idx):
    ''' function to align all the valid trials to the same t_cue '''
    min_cue_idx = np.min(start_cue_idx)
    max_cue_idx = np.max(start_cue_idx)
    time_to_add = max_cue_idx - min_cue_idx
    time_extent = np.zeros((all_times_array.shape[0], (all_times_array.shape[1] + time_to_add)))
    time_extent.fill(np.NaN)
    psth_extent = np.zeros((psth_array.shape[0], psth_array.shape[1], (psth_array.shape[2] + time_to_add)))
    psth_extent.fill(np.NaN)
    start_sample_new = np.zeros((start_cue_idx.shape[0]))
    start_delay_new = np.zeros((start_cue_idx.shape[0]))
    start_cue_new = np.zeros((start_cue_idx.shape[0]))
    for trial in range(start_cue_idx.shape[0]):
        start = int(np.abs(start_cue_idx[trial] - max_cue_idx))
        end = all_times_array.shape[1] + int(np.abs(start_cue_idx[trial] - max_cue_idx))
        time_extent[trial][start:end] = all_times_array[trial]
        psth_extent[:, trial, start:end] = psth_array[:, trial]
        start_sample_new[trial] = start_sample_idx[trial] + start
        start_delay_new[trial] = start_delay_idx[trial] + start
        start_cue_new[trial] = start_cue_idx[trial] + start
    return time_extent, psth_extent, start_sample_new, start_delay_new, start_cue_new

def extract_common_range(vaild_trials_array):
    start = np.max(vaild_trials_array[:,0])
    end = np.min(vaild_trials_array[:,1])
    return start, end

def analyze_common_range_RS(vaild_trials_array, idx_RS, psth_array, all_times_array, start_sample_array, start_delay_array, start_cue_array, start_sample_idx, start_delay_idx, start_cue_idx, all_trial_types, all_photo_stim_trials):
    # common range RS neurons (excitatory)
    start, end = extract_common_range(vaild_trials_array[idx_RS,:])
    
    # (n_units, n_common_trials, time_steps)
    psth_common = psth_array[idx_RS, start:end, :]
    
    # (n_common_trials, time_steps)
    times_common = all_times_array[start:end, :]
    
    # (n_common_trials, )
    start_sample_common = start_sample_array[start:end]
    start_delay_common = start_delay_array[start:end]
    start_cue_common = start_cue_array[start:end]
    start_sample_idx_common = start_sample_idx[start:end]
    start_delay_idx_common = start_delay_idx[start:end]
    start_cue_idx_common = start_cue_idx[start:end]
    common_trial_types = all_trial_types[start:end]
    common_photo_stim_trials = all_photo_stim_trials[start:end]
    return psth_common, times_common, start_sample_common, start_delay_common, start_cue_common, start_sample_idx_common, start_delay_idx_common, start_cue_idx_common, common_trial_types, common_photo_stim_trials

def extract_trial(trial_type, photo_stim_trial, type_num, ph_num, psth, times, start_sample, start_delay, start_cue, start_sample_idx, start_delay_idx, start_cue_idx):
    mask = (trial_type == type_num) & (photo_stim_trial == ph_num)
    psth_masked = psth[:, mask, :]
    times_masked = times[mask, :]
    start_sample_masked = start_sample[mask]
    start_delay_masked = start_delay[mask]
    start_cue_masked = start_cue[mask]
    start_sample_idx_masked = start_sample_idx[mask]
    start_delay_idx_masked = start_delay_idx[mask]
    start_cue_idx_masked = start_cue_idx[mask]
    return psth_masked, times_masked, start_sample_masked, start_delay_masked, start_cue_masked, start_sample_idx_masked, start_delay_idx_masked, start_cue_idx_masked

def prepare_selectivity(psth, start_delay_idx, start_cue_idx):
    mean_act_delay = np.zeros((psth.shape[0], start_delay_idx.shape[0]))
    for trial in range(start_delay_idx.shape[0]):
        # mean activity of each neuron in delay epoch
        mean_act_delay[:, trial] = np.nanmean(psth[:, trial, int(start_delay_idx[trial]):int(start_cue_idx[trial])], axis=1)
    
    # mean across trials for each neuron
    mean_across_trials = np.nanmean(mean_act_delay, axis=1)
    return mean_act_delay, mean_across_trials

def calculate_selectivity(psth_right_c, start_delay_idx_right_c, start_cue_idx_right_c, psth_left_c, start_delay_idx_left_c, start_cue_idx_left_c):
    ''' calculate the selectivity of each neuron '''
    mean_act_delay_right_c, mean_across_trials_right_c = prepare_selectivity(psth_right_c, start_delay_idx_right_c, start_cue_idx_right_c)
    mean_act_delay_left_c, mean_across_trials_left_c = prepare_selectivity(psth_left_c, start_delay_idx_left_c, start_cue_idx_left_c)
    
    # selectivity calculation
    sel_idx = (mean_across_trials_right_c - mean_across_trials_left_c) / (mean_across_trials_right_c + mean_across_trials_left_c)
    
    p_values = []
    for neuron in range(sel_idx.shape[0]):
        data_right = mean_act_delay_right_c[neuron]
        clean_right = data_right[~np.isnan(data_right)]
        data_left = mean_act_delay_left_c[neuron]
        clean_left = data_left[~np.isnan(data_left)]
        # p_values extraction
        p_values.append(ranksums(clean_right, clean_left)[1])
    
    selectivity = []
    
    for n, (sel, p) in enumerate(zip(sel_idx, np.array(p_values))):
        if sel > 0 and p < 0.05:
            selectivity.append("right")
        elif sel < 0 and p < 0.05:
            selectivity.append("left")
        else:
            selectivity.append("none")
    return selectivity