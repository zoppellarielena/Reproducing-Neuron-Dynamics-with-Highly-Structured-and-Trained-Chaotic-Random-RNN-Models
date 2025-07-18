import numpy as np
''' written by Elena Zoppellari'''

def normalize(R):
    return (R - R.min(axis=1, keepdims=True)) / (R.max(axis=1, keepdims=True) - R.min(axis=1, keepdims=True))
    
def normalize_jointly(R, L):
    # Combine target_R and target_L along axis 1 (side by side)
    combined = np.concatenate((R, L), axis=1)
    
    # Normalize across the combined data
    normalized_combined = (combined - combined.min(axis=1, keepdims=True)) / (combined.max(axis=1, keepdims=True) - combined.min(axis=1, keepdims=True))
    
    # Split the normalized data back into target_R and target_L
    normalized_R = normalized_combined[:, :R.shape[1]]
    normalized_L = normalized_combined[:, R.shape[1]:]
    
    return normalized_R, normalized_L

def t_com(normalized_data):
    return np.array([sum([normalized_data[i][t]*t for t in range(6000)])/sum([normalized_data[i][t] for t in range(6000)]) for i in range(normalized_data.shape[0])])

def realign_using_tcom(R_true, t_coms):
    R_aligned = np.zeros((R_true.shape[0], 2*R_true.shape[1]))
    for i in range(R_true.shape[0]):
        for t in range(6000):
            R_aligned[i][t-int(t_coms[i]+6000)] = R_true[i][t]
    return R_aligned

def R_ave_shift(R_aligned, t_coms_single):
    R_tcom = np.zeros((R_aligned.shape[0]))
    for t in range(R_aligned.shape[0]):
        if (t+int(t_coms_single)-3000 > 0) and (t+int(t_coms_single)-3000 < R_aligned.shape[0]):
            R_tcom[t+int(t_coms_single)-3000] = R_aligned[t]
    return R_tcom

def padding_data(R):
    new_R = np.zeros((2*R.shape[0]))
    for t in range(R.shape[0]):
        new_R[t+3000] = R[t]
    return new_R

def R_avg(aligned_data):
    return np.mean(aligned_data, axis=0)
    
def bVar(R, R_ave, t_coms):
    nom = sum([sum(np.abs(padding_data(R[i]) - R_ave_shift(R_ave, t_coms[i]))[3000:9000]) for i in range(R.shape[0])])**2
    R_mean = np.mean(R, axis=0)
    den = sum([sum(np.abs(padding_data(R[i]) - padding_data(R_mean))[3000:9000]) for i in range(R.shape[0])])**2
    return 1 - nom/den