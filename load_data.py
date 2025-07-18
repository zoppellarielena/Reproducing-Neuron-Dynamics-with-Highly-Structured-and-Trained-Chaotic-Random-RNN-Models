import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from matrix_analysis import trial_matrix
from scipy.io import loadmat
from matrix_analysis import calculate_selectivity, extract_trial, calc_idx, align_cue
''' written by Elena Zoppellari'''

# session names 
names = ['HI124_020617_units', 'HI124_020717_units', 'HI124_020817_units', 'HI124_020917_units', 'HI125_022817_units',
        'HI125_030117_units', 'HI125_030217_units', 'HI125_030317_units', 'HI126_032117_units', 'HI127_031317_units',
        'HI127_031417_units', 'HI127_031517_units', 'HI127_031617_units', 'HI129_040417_units', 'HI129_040517_units',
        'HI129_040617_units', 'HI129_040717_units', 'HI130_050817_units', 'HI130_050917_units', 'HI130_051217_units']

def extract_data(recording_index):
    filename = f'SiliconProbeData\FixedDelayTask\{names[recording_index]}.mat'
    data = loadmat(filename)
    units = data['unit'][0]
    # number of neurons recorded
    N_units = units.shape[0]
    psth = []
    times = []
    vaild_trials = []
    neuron_type = []
    
    for n_unit in tqdm(range(N_units)):
        psth_collection, bin_collection, trial_range, spikewidth = trial_matrix(n_unit, units)
        psth.append(psth_collection)
        times.append(bin_collection)
        vaild_trials.append(trial_range)
        neuron_type.append(spikewidth)

    return select_correct(units, psth, times, vaild_trials, neuron_type, N_units)
    

def select_correct(units, psth, times, vaild_trials, neuron_type, N_units):
    # same for all neurons
    start_sample = units[0]['Behavior']['Sample_start'][0][0].flatten()
    start_delay = units[0]['Behavior']['Delay_start'][0][0].flatten()
    start_cue = units[0]['Behavior']['Cue_start'][0][0].flatten()
    all_trial_types_0 = units[0]['Behavior']['Trial_types_of_response_vector'][0][0].flatten() # (n_trials, )
    all_photo_stim_trials_0 = units[0]['Behavior']['stim_trial_vector'][0][0].flatten() # (n_trials, )
    
    # remove all early licking and not-responsive trials
    good_trials = (all_trial_types_0 < 5)
    psth_array = np.array(psth)[:, good_trials, :]
    times_array = np.array(times)[:, good_trials, :]
    vaild_trials_array = np.array(vaild_trials) # n_neuron x 2
    neuron_type_array = np.array(neuron_type)   # n_neuron
    idx_RS = (neuron_type_array > 0.5)
    idx_FS = (neuron_type_array < 0.35)
    
    all_times_array = times_array[0,:,:]
    start_sample_array = start_sample[good_trials]
    start_delay_array = start_delay[good_trials]
    start_cue_array = start_cue[good_trials]
    all_trial_types = all_trial_types_0[good_trials]
    all_photo_stim_trials = all_photo_stim_trials_0[good_trials]
    
    start_sample_idx = calc_idx(all_times_array, start_sample_array)
    start_delay_idx = calc_idx(all_times_array, start_delay_array)
    start_cue_idx = calc_idx(all_times_array, start_cue_array)
    
    # all_times_array, psth_array, start_sample_idx, start_delay_idx, start_cue_idx = align_cue(start_cue_idx, all_times_array, psth_array, start_sample_idx, start_delay_idx)
    # analyze all trials, also the ones of FS neurons
    psth_right_c, times_right_c, start_sample_right_c, start_delay_right_c, start_cue_right_c, start_sample_idx_right_c, start_delay_idx_right_c, start_cue_idx_right_c = extract_trial(all_trial_types, all_photo_stim_trials, 1, 0, 
                                                                                                                                                                                psth_array, all_times_array, start_sample_array, 
                                                                                                                                                                                start_delay_array, start_cue_array, 
                                                                                                                                                                                start_sample_idx, start_delay_idx, 
                                                                                                                                                                                start_cue_idx)
    psth_left_c, times_left_c, start_sample_left_c, start_delay_left_c, start_cue_left_c, start_sample_idx_left_c, start_delay_idx_left_c, start_cue_idx_left_c = extract_trial(all_trial_types, all_photo_stim_trials, 2, 0, 
                                                                                                                                                                            psth_array, all_times_array, start_sample_array, 
                                                                                                                                                                            start_delay_array, start_cue_array, 
                                                                                                                                                                            start_sample_idx, start_delay_idx, 
                                                                                                                                                                            start_cue_idx)
    # selectivity according to the paper
    selectivity = calculate_selectivity(psth_right_c, start_delay_idx_right_c, start_cue_idx_right_c, psth_left_c, start_delay_idx_left_c, start_cue_idx_left_c)
    start_sample, start_delay, start_cue = extract_fixed(start_sample_idx_right_c, start_sample_idx_left_c, start_delay_idx_right_c, start_delay_idx_left_c, start_cue_idx_right_c, start_cue_idx_left_c)
    n_type = extract_type(idx_RS, idx_FS, N_units)
    
    return psth_right_c, psth_left_c, start_sample, start_delay, start_cue, selectivity, n_type

def extract_type(idx_RS, idx_FS, N_units):
    n_type = []
    for i in range(N_units):
        if idx_RS[i]:
            n_type.append("RS")
        elif idx_FS[i]:
            n_type.append("FS")
        else:
            n_type.append("other")
    return n_type

def extract_fixed(start_sample_idx_right_c, start_sample_idx_left_c, start_delay_idx_right_c, start_delay_idx_left_c, start_cue_idx_right_c, start_cue_idx_left_c):
    start_sample = int(np.median([np.median(start_sample_idx_right_c), np.median(start_sample_idx_left_c)]))
    start_delay = int(np.median([np.median(start_delay_idx_right_c), np.median(start_delay_idx_left_c)]))
    start_cue = int(np.median([np.median(start_cue_idx_right_c), np.median(start_cue_idx_left_c)]))
    return start_sample, start_delay, start_cue