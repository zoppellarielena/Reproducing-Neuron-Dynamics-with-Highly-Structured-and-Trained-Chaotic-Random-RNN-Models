import numpy as np
from scipy.special import logit, expit
from tqdm import tqdm as tqdm
'''
written by Elena Zoppellari
'''
class RandomNetwork(object):
    def __init__(self, N = 100, n_tasks = 1, tau = 10.0, tau_WN = 1000.0, T = 6000, mu = 0.0, g = 1.2, h0 = 10, theta=0, dt=1, thr=0.85, start_sample=1000, start_delay=2000, start_cue=4000, cut=False, J=None, frozen_input=None):
        '''
        params: 
        N = number of neurons
        n_tasks = number of tasks to teach to the network (left/right lick = 2 tasks)
        tau = integration time for the current dynamics
        tau_WN = integration time of the input
        T = number of time steps
        mu = mean of the gaussian from which the random RNN values are generated
        g = parameter to control the amount of randomness. g > 1 --> chaotic network
        h0 = input amplification
        dt = delta_t
        thr = convergence critiria based on pVar between predictions and targets
        start_sample = time step at which the sample starts
        start_delay = time step at which the delay starts
        start_cue = time stemp at which the cue starts
        cut = if True, remove the presample information
        J = synaptic matrix. If J = None, create a new one
        frozen_input = input frozen for each N. If frozen_input = None, create a new one
        '''
        self.N = N # number of neurons
        self.n_tasks = n_tasks # number of tasks
        self.tau = tau
        self.tau_WN = tau_WN
        end_times_to_remove = T - (start_cue + 250) # remove extra tails
        self.T = T - 1000 - end_times_to_remove # remove extra tails
        if cut:
            self.T = T - start_sample - end_times_to_remove
        self.dt = 1 # time step
        self.theta = theta
        self.h0 = h0
        # self.delay_time is used to decide the time window in which noise can be applied.
        # it can be the whole window (0, T) or only the first 600 time steps of delay
        self.delay_time = np.arange(0, self.T) # np.arange(start_delay-1000, (start_delay-1000+600)) # np.arange(0, self.T)
        self.start_delay = start_delay
        if cut:
            self.start_delay = start_delay - start_sample
        self.start_cue = start_cue
        self.thr = thr
        if cut:
            self.start_cue = start_cue - start_sample
        
        # build random interaction matrix
        var = g**2/N 
        if J is None:
            self.J = np.random.normal(mu, np.sqrt(var), size=(N,N)) 
        else:
            self.J = J

        # create frozen input
        if frozen_input is None:
            self.generate_frozen_input(start_sample, start_delay, start_cue, cut=cut)
        else:
            self.freezed_noise = frozen_input

    def train(self, target = None, n_iters=10, idx = None, p=1, test=False, perturbate=False, amp_noise=None):
        '''
        train the random network. Params:
        target = list of targets (must correspond with n_tasks)
        n_iters = number of max iterations
        idx = indices that represent neurons of subtasks
        test = active test modality, don't train but only test the network
        perturbate = if True, add noise to a time window
        amp_noise = amplitude of noise. valid id perturbate = True
        '''
        initial_J = self.J.copy()
        self.target = target
        if self.target is None:
            raise ValueError("target must be defined for training.")
        self.idx = idx # indices that represent neurons of subtasks (dim=-1, common neurons)
        if self.idx is None:
            raise ValueError("self.idx must be defined for training.")
        self.p = p # fraction of trainable neurons
        Np = int(self.N*p)
        N_train = np.random.choice(self.N, size=Np, replace=False)
        self.N_train = N_train[np.argsort(N_train)]
        
        # initialize variables
        self.x = [np.zeros(self.N) for i in range(self.n_tasks)] # init with all 0s for reproducibility
        self.r = [self.phi(self.x[i], self.theta) for i in range(self.n_tasks)] # init firing rates
        #self.r_history = [np.zeros((self.T, n_iters, self.N)) for i in range(self.n_tasks)]
        self.pvar_history = [np.zeros((n_iters)) for i in range(self.n_tasks)] # init collection of pVar
        self.P = self.h0 * np.eye(self.N) # init P matrix
        self.r_history = [np.zeros((self.T, self.N)) for i in range(self.n_tasks)] # init collection of firing rates
        
        # find R from data (synthetic)
        self.R_true = [self.f(self.target[i]) for i in range(self.n_tasks)]
        
        if test:
            for i in range(self.n_tasks):
                if perturbate:
                    arr_noise = amp_noise * np.sqrt(1 / (2*self.tau)) * np.random.normal(loc=0, scale=1, size=(self.T))
                    self.evolve_choice(0, train=False, task_idx=i, perturbate=True, arr_noise=arr_noise)
                else:
                    self.evolve_choice(0, train=False, task_idx=i, perturbate=False, arr_noise=None)
                
                self.pvar_history[i][0] = self.pVar(self.target[i], self.r_history[i].T)
            return self.r_history, self.pvar_history, self.J, initial_J
        else:
            # train
            for iter in tqdm(range(n_iters)):
                for i in range(self.n_tasks):
                    self.r_history[i][0, :] = self.r[i].copy()
                # train
                for i in range(self.n_tasks):
                    self.evolve_choice(iter, train=True, task_idx=i, perturbate=False, arr_noise=None)
                # test
                for i in range(self.n_tasks):
                    if perturbate:
                        arr_noise = amp_noise * np.sqrt(1 / (2*self.tau)) * np.random.normal(loc=0, scale=1, size=(self.T))
                        self.evolve_choice(iter, train=False, task_idx=i, perturbate=True, arr_noise=arr_noise)
                    else:
                        self.evolve_choice(iter, train=False, task_idx=i, perturbate=False, arr_noise=None)
                        
                # calculate pVar
                for i in range(self.n_tasks):
                    self.pvar_history[i][iter] = self.pVar(self.target[i], self.r_history[i].T)
                # early end if all predicted sequences have a pVar > thr compared with the correspective target
                if all(self.pvar_history[i][iter] > self.thr for i in range(self.n_tasks)):
                    print(f"convergence reached at iter={iter}")
                    mean_error = [np.mean(np.abs(np.dot(self.J, self.r_history[i].T) - self.target[i])) for i in range(self.n_tasks)]
                    print(f"iter {iter} p_var={[np.round(self.pvar_history[i][iter]*100,2) for i in range(self.n_tasks)]}% and avg_error={[np.round(mean_error[i],2) for i in range(self.n_tasks)]}")
                    break
                if ((iter%50 == 0) and (iter > 0)) or (iter==(n_iters-1)):
                    mean_error = [np.mean(np.abs(np.dot(self.J, self.r_history[i].T) - self.target[i])) for i in range(self.n_tasks)]
                    print(f"iter {iter} p_var={[np.round(self.pvar_history[i][iter]*100,2) for i in range(self.n_tasks)]}% and avg_error={[np.round(mean_error[i],2) for i in range(self.n_tasks)]}")
        return self.r_history, self.pvar_history, self.J, initial_J, self.freezed_noise

    # activation function
    def phi(self, x, theta):
        return expit(x - theta)
        
    # target function
    def f(self, R):
        eps = 1e-5
        return logit(np.clip(R, eps, 1 - eps))

    def c(self, r, Pr):
        return 1/(1 + np.dot(r, Pr))

    def pVar(self, R, r):
        nom = sum([sum(np.abs(R[i] - r[i])) for i in range(R.shape[0])])**2
        R_mean = np.mean(R, axis=0)
        den = sum([sum(np.abs(R[i] - R_mean)) for i in range(R.shape[0])])**2
        return 1 - nom/den


    def evolve_choice(self, iter, train=False, task_idx=0, perturbate=False, arr_noise=None):
        '''
        evolve the dynamics and if train = True, update the synaptic weights. params:
        iter = interation step
        train = if True, train the network
        task_idx = task idx
        perturbate  if True, add noise
        arr_noise = noise to add if perturbate = True
        '''
        for step in range(1, self.T):
            # 1. the dynamics evolve from t-dt to t
            t = step * self.dt
            x = self.x
            r = self.r
            i = task_idx 
            # Euler update for x
            if (perturbate == True) and (train == False) and (step in self.delay_time):
                dx = (-x[i] + np.dot(self.J, r[i]) + self.freezed_noise[i][:, step] + arr_noise[step]) / self.tau
                x_out = x[i] + self.dt * dx # x(t)
                x[i] = x_out
                
            else:
                dx = (-x[i] + np.dot(self.J, r[i]) + self.freezed_noise[i][:, step]) / self.tau
                x_out = x[i] + self.dt * dx # x(t)
                x[i] = x_out
            
            # Compute r_i(t) from activation variable x(t)
            r_out = self.phi(x_out, self.theta) # r(t)
            r[i] = r_out
            if not train:
                self.r_history[i][step, :] = r_out
    
            # error function
            e = np.dot(self.J, r_out) - self.R_true[i][:, step]

            if train: # FORCE learning
                # P update
                Pr = np.dot(self.P, r_out)
                P_updated = self.P - np.outer(Pr, Pr.T) / (1 + np.dot(r[i], Pr))
                Pr_updated = np.dot(P_updated, r[i]) 
                self.P = P_updated
                
                # J update
                J_new = self.J        
                mask_j = np.array([j in self.N_train for j in range(self.N)])
                
                mask_i = np.array([k in self.idx[i] for k in range(self.N)])
                Pr = np.dot(P_updated, r_out)
                cval = self.c(r_out, Pr)
                
                outer = np.outer(e[mask_i], P_updated[np.ix_(mask_j, mask_j)] @ r_out[mask_j])
                J_new[np.ix_(mask_i, mask_j)] -= cval * outer

                self.J = J_new
                

    def generate_frozen_input(self, start_sample, start_delay, start_cue, cut=False):
        '''
        generate the input signal. For each N, this signal is fixed across all learning iterations. params:
        start_sample = time step at which the mice hear the signal
        start_delay = time step at which the delay starts
        start_cue = time step at which the cue signal is given and the mice can move

        the noise is generated following a Ornstein–Uhlenbeck process. The evolution is:
        - common for signal before the sample cue (the activity of the neuron is at rest)
        - different for left and right sounds during sample cue (they are hearing a different sound)
        - common during the delay period (mice are not hearing sounds, they have to keep the memory collected in sample cue)
        - different for left and right sounds after cue signal 
        
        '''
        sqrt_dt = np.sqrt(1)
        sqrt_tau = np.sqrt(self.tau_WN)
        freezed_noise = [np.zeros((self.N, self.T)) for i in range(self.n_tasks)]
        common_stimulus = np.zeros((self.N, self.T))
        
        # Initialize at 1
        for i in range(self.n_tasks):
            freezed_noise[i][:, 0] = 0.5
            
        common_stimulus[:, 0] = 0.5
        arr_normal_values_common = np.random.normal(loc=0.0, scale=1, size=(self.N, self.T))
        arr_normal_values = [np.random.normal(loc=0.0, scale=0.5, size=(self.N, self.T)) for i in range(self.n_tasks)]

        if cut:
            for idx_t in range(1, start_delay-start_sample):
                h = common_stimulus[:, idx_t - 1]
                common_stimulus[:, idx_t] = h + 1 * (-h + self.h0 * arr_normal_values_common[:, idx_t]) / self.tau_WN
                for i in range(self.n_tasks):
                    h_i = freezed_noise[i][:, idx_t - 1]
                    freezed_noise[i][:, idx_t] = h_i + 1 * (-h_i + self.h0 * arr_normal_values[i][:, idx_t]) / self.tau_WN
            for idx_t in range(start_delay-start_sample, start_cue-start_sample):
                h = common_stimulus[:, idx_t - 1]
                common_evolve = h + 1 * (-h + self.h0 * arr_normal_values_common[:, idx_t]) / self.tau_WN
                common_stimulus[:, idx_t] = common_evolve.copy()
                for i in range(self.n_tasks):
                    freezed_noise[i][:, idx_t] = common_evolve.copy()
            for idx_t in range(start_cue-start_sample, self.T):
                h = common_stimulus[:, idx_t - 1]
                common_stimulus[:, idx_t] = h + 1 * (-h + self.h0 * arr_normal_values_common[:, idx_t]) / self.tau_WN
                for i in range(self.n_tasks):
                    h_i = freezed_noise[i][:, idx_t - 1]
                    freezed_noise[i][:, idx_t] = h_i + 1 * (-h_i + self.h0 * arr_normal_values[i][:, idx_t]) / self.tau_WN
        else:
            for idx_t in range(1, start_sample-1000):
                h = common_stimulus[:, idx_t - 1]
                common_evolve = h + 1 * (-h + self.h0 * arr_normal_values_common[:, idx_t]) / self.tau_WN
                common_stimulus[:, idx_t] = common_evolve.copy()
                for i in range(self.n_tasks):
                    freezed_noise[i][:, idx_t] = common_evolve.copy()
            for idx_t in range(start_sample-1000, start_delay-1000):
                h = common_stimulus[:, idx_t - 1]
                common_stimulus[:, idx_t] = h + 1 * (-h + self.h0 * arr_normal_values_common[:, idx_t]) / self.tau_WN
                for i in range(self.n_tasks):
                    h_i = freezed_noise[i][:, idx_t - 1]
                    freezed_noise[i][:, idx_t] = h_i + 1 * (-h_i + self.h0 * arr_normal_values[i][:, idx_t]) / self.tau_WN
            for idx_t in range(start_delay-1000, start_cue-1000):
                h = common_stimulus[:, idx_t - 1]
                common_evolve = h + 1 * (-h + self.h0 * arr_normal_values_common[:, idx_t]) / self.tau_WN
                common_stimulus[:, idx_t] = common_evolve.copy()
                for i in range(self.n_tasks):
                    freezed_noise[i][:, idx_t] = common_evolve.copy()
            for idx_t in range(start_cue-1000, self.T):
                h = common_stimulus[:, idx_t - 1]
                common_stimulus[:, idx_t] = h + 1 * (-h + self.h0 * arr_normal_values_common[:, idx_t]) / self.tau_WN
                for i in range(self.n_tasks):
                    h_i = freezed_noise[i][:, idx_t - 1]
                    freezed_noise[i][:, idx_t] = h_i + 1 * (-h_i + self.h0 * arr_normal_values[i][:, idx_t]) / self.tau_WN
            
        self.freezed_noise = freezed_noise