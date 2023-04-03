from scipy import special
import torch
import numpy as np

import tqdm

from src import tools

# This module runs numerical experiments for the logarithmic
# Bayesian model using idealized measurement data.

class Bayesian_Optimal():

    # This class represents the Bayesian model with optimal performance
    # on the error detection task.

    def __init__(self, depth):

        # The depth attribute determines how many prior syndrome measurements to
        # condition on when calculating the Gaussian likelihood of a new syndrome
        # measurement. For depth > 0, this allows the model to leverage any 
        # autocorrelation that may be present in the data.

        self.depth = depth

        # This is simply a list of the other attributes that will be used.

        self.transition_matrix = None
        self.gauss_means = None
        self.gauss_precs = None
        self.prior = None
        self.prev_synd = None
        self.time_step = None

    def perfect_train(self, gamma_m, eta, error_rate, time_step, num_samples = 1000):

        # This method calculates ideal transition matrices and Gaussian
        # precisions for the simulated data using the parameters provided.
        # The depth must be set to 0 when using this training scheme. 

        assert self.depth == 0, "Depth must be 0 when using perfect_train."
        self.time_step = time_step
        self.num_parallel = int(1000000/num_samples)
        expon = torch.tensor(-2*error_rate*self.time_step)
        no_flip_prob = 0.5*(1 + torch.exp(expon))
        flip_prob = 0.5*(1 - torch.exp(expon))
        (hist_array, means) = create_combined_histograms(3*error_rate*time_step)
        self.hist_array = torch.tensor(hist_array).cuda()
        self.means = torch.tensor(means).cuda()
        self.transition_matrix = torch.zeros([8, 8], dtype = torch.float32).cuda()
        for i in range(8):
            for j in range(8):
                diff = i ^ j
                if diff == 0:
                    prob = no_flip_prob**3
                elif ((diff == 4) | (diff == 2) | (diff == 1)):
                    prob = flip_prob*(no_flip_prob**2)
                elif diff == 7:
                    prob = flip_prob**3
                else:
                    prob = (flip_prob**2)*no_flip_prob
                self.transition_matrix[i, j] = prob
        gauss_prec = (2 * gamma_m * eta) * self.time_step
        self.joint_precs = gauss_prec

    def tracking_test(self, test_data, labels, initial):

        # This method runs an error-tracking test of the Bayesian model on the passed 
        # test data, printing the progress and eventual accuracy while also returning a 
        # tensor of prior probabilities at each step. All runs are processed in parallel
        # For each run the prior probabililty is determineed by looking at the first label
        # of the trajectory.

        m_1 = test_data[:, :, 0]
        m_2 = test_data[:, :, 1]
        (num_runs, num_steps) = m_1.shape
        self.prior = torch.zeros([num_runs, 8]).cuda()
        self.prior[torch.arange(num_runs), initial.type(torch.long)] = 1
        num_correct = torch.zeros([num_runs]).cuda()
        for step in range(self.depth, num_steps):

            # For each step, the new prior is calculated by contracting the previous
            # prior with the transition matrix, then scaling by the measurement 
            # probabilities.

            m_1_slice = m_1[:, step - self.depth:step + 1]
            m_2_slice = m_2[:, step - self.depth:step + 1]
            gauss_matrices = self.get_gaussian_weights(m_1_slice, m_2_slice)
            weighted_matrices = gauss_matrices * self.transition_matrix
            unnorm_probs = torch.einsum("al,alr->ar", self.prior.float(), weighted_matrices)
            self.prior = unnorm_probs / torch.sum(unnorm_probs, dim = 1, keepdim = True)
            predictions = torch.argmax(self.prior, dim = 1)
            true_states = labels[:, step]
            fidelity = (predictions == true_states)
            num_correct = num_correct + fidelity
            print("\rStep {}/{}".format(step, num_steps), end = "")

        acc = num_correct * 100 / (num_steps - self.depth)
        print("\nAcc: {:.2f}%".format(torch.mean(acc)))
        print(f"Fidelity: {torch.mean(fidelity.float() * 100):.2f}%")
        return fidelity.float().mean().item()

    def get_gaussian_weights(self, m_1_batch, m_2_batch):

        # This method calculates the Gaussian likelihoods for the syndrome pair in
        # the most recent step, conditioned on a number of prior measurements determined
        # by the depth attribute. This means that the m_1_batch and m_2_batch must both  
        # have shapes of (num_runs, depth + 1).

        batch_size = m_1_batch.shape[0]
        output_matrices = torch.zeros([batch_size, 8, 8]).cuda()
        joint_batch = torch.cat([m_1_batch, m_2_batch], dim = 1)
        start = 0
        for end in range(self.num_parallel, batch_size + self.num_parallel, self.num_parallel):
            centered = joint_batch[start:end, None, None] - self.means[None]
            joint_expon = -0.5*self.joint_precs*(centered**2).sum(-1) #torch.einsum("amnsi,ij,amnsj->amns", centered.float(), self.joint_precs.float(), centered.float())
            log_output = joint_expon + 0.5*np.math.log(self.joint_precs**2)
            samp_outputs = torch.exp(log_output) # Only exponentiate at the end
            weighted_samples = samp_outputs[:, None, None]*self.hist_array[None]
            output_matrices[start:end] = weighted_samples.sum((-2, -1))
            start = end
        return output_matrices

class Bayesian_Log():

    # This class represents the Bayesian model which approximates the
    # optimal model in an efficient manner using logarithmic probabilties.

    def __init__(self, depth, add_type):

        # The depth attribute determines how many prior syndrome measurements to
        # condition on when calculating the Gaussian likelihood of a new syndrome
        # measurement. For depth > 0, this allows the model to leverage any 
        # autocorrelation that may be present in the data.

        self.depth = depth
        self.add_type = add_type
        if add_type not in ["exact", "half_exact", "half_lookup", "max"]:
            raise ValueError(f"Addition type {add_type} not valid.")

        # This is simply a list of the other attributes that will be used.

        self.transition_matrix = None
        self.gauss_means = None
        self.gauss_precs = None
        self.prior = None
        self.prev_synd = None
        self.add_func = get_log_add_func(5, -4)

    def perfect_train(self, gamma_m, eta, error_rate, time_step):

        # This method calculates ideal transition matrices and Gaussian
        # precisions for the simulated data using the parameters provided.
        # The depth must be set to 0 when using this training scheme. 

        assert self.depth == 0, "Depth must be 0 when using perfect_train."
        expon = torch.tensor(-2*error_rate*time_step)
        no_flip_prob = 0.5*(1 + torch.exp(expon))
        flip_prob = 0.5*(1 - torch.exp(expon))
        self.transition_matrix = torch.zeros([8, 8], dtype = torch.float32)
        for i in range(8):
            for j in range(8):
                diff = i ^ j
                if diff == 0:
                    prob = no_flip_prob**3
                elif ((diff == 4) | (diff == 2) | (diff == 1)):
                    prob = flip_prob*(no_flip_prob**2)
                elif diff == 7:
                    prob = flip_prob**3
                else:
                    prob = (flip_prob**2)*no_flip_prob
                self.transition_matrix[i, j] = prob
        gauss_prec = (2 * gamma_m * eta) * time_step
        self.joint_means = torch.tensor([[1, 1], [1,-1], [-1, -1], [-1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
        self.joint_precs = gauss_prec * torch.eye(2).unsqueeze(0).expand(8, -1, -1)
        self.norm_factor = get_normalization_factor(gauss_prec, 2)

    def tracking_test(self, test_data, labels, initial):

        # This method runs an error-tracking test of the Bayesian model on the passed 
        # test data, printing the progress and eventual accuracy while also returning a 
        # tensor of prior probabilities at each step. All runs are processed in parallel
        # For each run the prior probabililty is determineed by looking at the first label
        # of the trajectory.

        m_1 = test_data[:, :, 0]
        m_2 = test_data[:, :, 1]
        (num_runs, num_steps) = m_1.shape
        self.log_prior = torch.full([num_runs, 8], -20.0)
        self.log_prior[torch.arange(num_runs), initial.type(torch.long)] = 0.0
        num_correct = torch.zeros([num_runs])
        norms = []
        stds = []
        for step in range(self.depth, num_steps):

            # For each step, the new prior is calculated by contracting the previous
            # prior with the transition matrix, then scaling by the measurement 
            # probabilities.

            m_1_slice = m_1[:, step - self.depth:step + 1]
            m_2_slice = m_2[:, step - self.depth:step + 1]
            log_likelihood = self.get_gaussian_outputs(m_1_slice, m_2_slice)
            new_log_priors = self.get_log_markov(self.log_prior)
            unnorm_log_probs = log_likelihood + new_log_priors
            self.log_prior = unnorm_log_probs + self.norm_factor
            norm_max = torch.abs(self.log_prior).max(1)[0]
            norms.append(norm_max.mean().item())
            stds.append(torch.std(norm_max))
            predictions = torch.argmax(self.log_prior, dim = 1)
            true_states = labels[:, step]
            fidelity = (predictions == true_states)
            num_correct = num_correct + fidelity
            print("\rStep {}/{}".format(step, num_steps), end = "")

        acc = num_correct * 100 / (num_steps - self.depth)
        print("\nAcc: {:.2f}%".format(torch.mean(acc)))
        print(f"Fidelity: {torch.mean(fidelity.float() * 100):.2f}%")
        return fidelity.float().mean().item()

    def get_gaussian_outputs(self, m_1_batch, m_2_batch):

        # This method calculates the Gaussian likelihoods for the syndrome pair in
        # the most recent step, conditioned on a number of prior measurements determined
        # by the depth attribute. This means that the m_1_batch and m_2_batch must both  
        # have shapes of (num_runs, depth + 1).

        (num_runs, num_steps) = m_1_batch.shape
        joint_batch = torch.cat([m_1_batch, m_2_batch], dim = 1)
        marg_batch = torch.cat([m_1_batch[:, :-1], m_2_batch[:, :-1]], dim = 1)

        joint_centered = joint_batch.reshape([num_runs, 1, 2*(self.depth + 1)]) - self.joint_means.reshape([1, 8, 2*(self.depth + 1)])
        joint_expon = -0.5*torch.einsum("ali,lij,alj->al", joint_centered.float(), self.joint_precs.float(), joint_centered.float())
        joint_output = joint_expon #+ 0.5*torch.slogdet(self.joint_precs)[1]

        if self.depth > 0: # The marginal distribution only exists if depth > 0
            marg_batch = torch.cat([m_1_batch[:, :-1], m_2_batch[:, :-1]], dim = 1)
            marg_centered = marg_batch.reshape([num_runs, 1, 2*self.depth]) - self.marg_means.reshape([1, 8, 2*self.depth])
            marg_expon = -0.5*torch.einsum("ali,lij,alj->al", marg_centered.float(), self.marg_precs.float(), marg_centered.float())
            marg_output = marg_expon
            log_output = joint_output - marg_output
        else:
            log_output = joint_output
        return log_output

    def get_log_markov(self, log_priors):

        # This function computes the logsumexp of the
        # the prior probabilities combined with the 
        # Bayesian update from the measurements.

        log_combined = log_priors[:, :, None] + torch.log(self.transition_matrix)[None]
        if self.add_type == "exact":
            new_log_priors = torch.logsumexp(log_combined, 1)
        elif self.add_type == "max":
            new_log_priors = torch.max(log_combined, 1)[0]
        else:
            log_sorted = torch.sort(log_combined, 1)[0]
            if self.add_type == "half_exact":
                new_log_priors = self.add_logs(log_sorted[:, -1], log_sorted[:, -2])
            elif self.add_type == "half_lookup":
                new_log_priors = self.add_func(log_sorted[:, -1], log_sorted[:, -2])
        return new_log_priors

    def add_logs(self, *vals):

        # This computes the logsumexp on a only a portion of 
        # the values.

        num_vals = len(vals)
        assert num_vals >= 1
        log_sorted = torch.sort(torch.stack(vals), 0)[0]
        cuml_sum = log_sorted[-1]
        for i in range(1, num_vals):
            cuml_sum = cuml_sum + torch.log(1 + torch.exp(log_sorted[-(i + 1)] - cuml_sum))
        return cuml_sum

class Bayesian_Linear():

    # This class represents the best linear approximation of the optimal
    # Bayesian model.

    def __init__(self, depth):

        # The depth attribute determines how many prior syndrome measurements to
        # condition on when calculating the Gaussian likelihood of a new syndrome
        # measurement. For depth > 0, this allows the model to leverage any 
        # autocorrelation that may be present in the data.

        self.depth = depth

        # This is simply a list of the other attributes that will be used.

        self.transition_matrix = None
        self.gauss_means = None
        self.gauss_precs = None
        self.prior = None
        self.prev_synd = None

    def perfect_train(self, gamma_m, eta, error_rate, time_step):

        # This method calculates ideal transition matrices and Gaussian
        # precisions for the simulated data using the parameters provided.
        # The depth must be set to 0 when using this training scheme. 

        self.time_step = time_step
        self.rate_matrix = torch.zeros([8, 8]).float()
        for row in range(8):
            for flip in [4, 2, 1]:
                col = row ^ flip
                self.rate_matrix[row, col] = error_rate

        self.joint_covs =  1/(2 * gamma_m * eta * time_step) * torch.eye(2)[None].repeat(8, 1, 1)
        self.joint_means = torch.tensor([[1, 1], [1,-1], [-1, -1], [-1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])

    def tracking_test(self, test_data, labels, initial):

        # This method runs an error-tracking test of the Bayesian model on the passed 
        # test data, printing the progress and eventual accuracy while also returning a 
        # tensor of prior probabilities at each step. All runs are processed in parallel
        # For each run the prior probabililty is determineed by looking at the first label
        # of the trajectory.

        m_1 = test_data[:, :, 0]
        m_2 = test_data[:, :, 1]
        norms = []
        stds = []
        (num_runs, num_steps) = m_1.shape
        self.prior = torch.zeros([num_runs, 8])
        self.prior[torch.arange(num_runs), initial.type(torch.long)] = 1
        num_correct = torch.zeros([num_runs])
        for step in range(self.depth, num_steps):

            # For each step, the new prior is calculated by contracting the previous
            # prior with the transition matrix, then scaling by the measurement 
            # probabilities.

            m_1_slice = m_1[:, step - self.depth:step + 1]
            m_2_slice = m_2[:, step - self.depth:step + 1]
            likelihood = self.get_gaussian_outputs(m_1_slice, m_2_slice)
            derivative_matrix = likelihood + self.rate_matrix
            slope = torch.einsum("nij,nj->ni", derivative_matrix, self.prior)
            unnorm_probs = self.prior + self.time_step * slope
            self.prior = unnorm_probs / torch.sum(unnorm_probs, dim = 1, keepdim = True)
            predictions = torch.argmax(self.prior, dim = 1)
            true_states = labels[:, step]
            fidelity = (predictions == true_states)
            num_correct = num_correct + fidelity
            norm_max = torch.abs(self.prior).max(1)[0]
            norms.append(norm_max.mean().item())
            stds.append(torch.std(norm_max))
            print("\rStep {}/{}".format(step, num_steps), end = "")

        acc = num_correct * 100 / (num_steps - self.depth)
        print("\nAcc: {:.2f}%".format(torch.mean(acc)))
        print(f"Fidelity: {torch.mean(fidelity.float() * 100):.2f}%")
        # np.save("paper/linear_mean", norms)
        # np.save("paper/linear_stds", stds)
        return fidelity.float().mean().item()

    def get_gaussian_outputs(self, m_1_batch, m_2_batch):

        # This method calculates the Gaussian likelihoods for the syndrome pair in
        # the most recent step, conditioned on a number of prior measurements determined
        # by the depth attribute. This means that the m_1_batch and m_2_batch must both  
        # have shapes of (num_runs, depth + 1).

        (num_runs, num_steps) = m_1_batch.shape
        joint_batch = torch.cat([m_1_batch, m_2_batch], dim = 1)
        output_pairs = joint_batch[:, None] * self.joint_means[None] / torch.diagonal(self.joint_covs*self.time_step, 0, 1, 2)[None]
        outputs = torch.diag_embed(output_pairs.sum(-1))
        return outputs

class DoubleThreshold():

    # This class represents an efficient implementation of the double-threshold
    # algorithm that is commonly used for error correction.

    def __init__(self, tau, Gamma_m=2.5e6, dt=32e-9, upper_th=0.8, lower_th=-0.54, inverted=False):

        # The tau attribute sets the filtering rate, while gamma_m sets the error rate
        # and dt sets the time increment. upper_th and lower_th set the upper and lower
        # bounds for the detection thresholds.

        self.tau = tau
        self.upper_th = upper_th
        self.lower_th = lower_th
        self.Gamma_m = Gamma_m
        self.dt = dt
        self.state_table = {0: (0, 7), 1: (4, 3), 2: (2, 5), 3: (1, 6)}
        self.start_mean_sim = {0: [1, 1], 1: [1, -1], 2: [-1, -1], 3: [-1, 1], 4: [-1, 1], 5: [-1, -1], 6: [1, -1], 7: [1, 1]}
        self.inverted = inverted

    def filter(self, dQ1, dQ2, labels):

        # This function takes in measurement values through dQ1 and dQ2
        # and then applies an exponentially decaying filter to smooth out
        # the signal.

        r = 1 / self.tau
        prefactor = 1 - self.dt * r
        NInverse = r
        num_trajs, num_steps = dQ1.shape[0], dQ1.shape[1]
        R1 = np.zeros((num_trajs, num_steps+1))
        R2 = np.zeros((num_trajs, num_steps+1))

        if not self.inverted:
            print('Alert: Need to change the start mean sim if it is the experimental data since the means are inverted')
        else:
            for key, _ in self.start_mean_sim.items():
                self.start_mean_sim[key] = [-1 * i for i in self.start_mean_sim[key]]

        for i in range(num_trajs):
            R1[i, 0] = self.start_mean_sim[labels[i, 0]][0]
            R2[i, 0] = self.start_mean_sim[labels[i, 0]][1]

        for t_index in range(num_steps):
            R1[:, t_index+1] = np.real(prefactor * R1[:, t_index] + dQ1[:, t_index] * NInverse)
            R2[:, t_index+1] = np.real(prefactor * R2[:, t_index] + dQ2[:, t_index] * NInverse)

        R1, R2 = R1[:, 1:], R2[:, 1:]
        return R1, R2

    def decide_curr_state(self, possible_state1, possible_state2, prev_state):
        
        # This function chooses the state that is only zero or one bitflip away from the 
        # previous state. The zero bitflip case applies to the integrated signals in the
        # first few steps when the integrated signal passes through the threshold for the 
        # first time.
        
        if int(prev_state) ^ possible_state1 in [0, 1, 2, 4]:
            return possible_state1
        elif int(prev_state) ^ possible_state2 in [0, 1, 2, 4]:
            return possible_state2
        else:
            raise 'What happened?'

    def identify_subspace(self, R1, R2, initial_states):
        
        # This function identifies the most likely configuration of thw qubits, with 
        # b1 and b2 serving as the thresholds that determine the assignment using signals
        # R1 and R2 respectively.
        
        num_trajs, num_steps = R1.shape[0], R1.shape[1]
        pred_states = np.zeros((num_trajs, num_steps))
        pred_states[:, 0] = initial_states

        for traj in tqdm(range(num_trajs), leave=True, position=0):
            for t in np.arange(1, num_steps):
                prev_state = pred_states[traj, t-1]
                R1_curr, R2_curr = R1[traj, t], R2[traj, t]

                if not self.inverted:
                    if R1_curr > self.upper_th and R2_curr > self.upper_th:
                        subspace = 0
                    elif R1_curr < self.lower_th and R2_curr > self.upper_th:
                        subspace = 1
                    elif R1_curr < self.lower_th and R2_curr < self.lower_th:
                        subspace = 2
                    elif R1_curr > self.upper_th and R2_curr < self.lower_th:
                        subspace = 3
                    else:
                        curr_state = prev_state
                        pred_states[traj, t] = curr_state
                        continue
                else:
                    if R1_curr < self.lower_th and R2_curr < self.lower_th:
                        subspace = 0
                    elif R1_curr > self.upper_th and R2_curr < self.lower_th:
                        subspace = 1
                    elif R1_curr > self.upper_th and R2_curr > self.upper_th:
                        subspace = 2
                    elif R1_curr < self.lower_th and R2_curr > self.upper_th:
                        subspace = 3
                    else:
                        curr_state = prev_state
                        pred_states[traj, t] = curr_state
                        continue

                possible_state1, possible_state2 = self.state_table[subspace]
                curr_state = self.decide_curr_state(possible_state1, possible_state2, prev_state)
                pred_states[traj, t] = curr_state
            
        return pred_states

def generate_mean_histograms(max_errors, num_intervals, num_samples):

    # This function is used to generate histograms for the syndrome densities,
    # which then serve as a representation of the syndrome distribution in the 
    # optimal Bayesian model.

    histos = np.zeros([max_errors + 1, max_errors + 1, max_errors + 1, num_intervals, num_intervals])
    num_errors = np.zeros([max_errors + 1, max_errors + 1, max_errors + 1])
    for i in range(max_errors + 1):
        for j in range(max_errors + 1 - i):
            for k in range(max_errors + 1 - i - j):
                print(f"{i} {j} {k}", end = "\r")
                means = tools.get_partial_means(num_samples, (i, j, k)).numpy()
                hist = np.histogram2d(means[:, 0], means[:, 1], num_intervals, range = [[-1, 1], [-1, 1]])[0]
                mass = hist / num_samples
                histos[i, j, k] = mass
                num_errors[i, j, k] = i + j + k
    print("")
    return (histos, num_errors)

def create_combined_histograms(mean_errors):

    # This function generates histograms for the measurement signal,
    # which is a combination of the syndrome mean distribution and the 
    # Gaussian noise.

    synd_means = np.array([[1, 1], [1,-1], [-1, -1], [-1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    hist_data = np.load("histos_100.npz")
    histos = hist_data["hist"]
    num_errors = hist_data["num"]
    bit_rep = np.unpackbits(np.arange(8, dtype = "uint8")[:, None], 1, 8).astype("int")[:, 5:]
    bit_diff_matrix = np.abs(bit_rep[None] - bit_rep[:, None])
    (X, Y) = np.meshgrid(np.linspace(-1, 1, histos.shape[-1]), np.linspace(-1, 1, histos.shape[-1]))
    hist_array = np.zeros((8, 8) + histos.shape[-2:])
    for i in range(8):
        for j in range(8):
            bit_diff = bit_diff_matrix[i, j]
            hists = histos[bit_diff[0]::2, bit_diff[1]::2, bit_diff[2]::2]
            errors = num_errors[bit_diff[0]::2, bit_diff[1]::2, bit_diff[2]::2]
            poisson_unorm = np.exp(-mean_errors)*(mean_errors**errors)/special.factorial(errors)
            weighted_hists = hists*poisson_unorm[:, :, :, None, None]
            combined_hist = weighted_hists.sum((0, 1, 2))
            if synd_means[i][0] == -1:
                combined_hist = np.flip(combined_hist, 0)
            if synd_means[i][1] == -1:
                combined_hist = np.flip(combined_hist, 1)
            hist_array[i, j] = combined_hist / combined_hist.sum()
    means = np.stack([Y, X], axis = 2)
    return (hist_array, means)

def get_log_add_func(num_segments, zero_cutoff):

    # This function simulates using a lookup table to 
    # compute the logsumexp rather than using floating-point
    # operations.

    increments = torch.linspace(zero_cutoff, 0, num_segments)
    values = torch.log2(1 + 2**increments)
    def add_func(a, b):
        diff = b - a
        closest = torch.argmin(torch.abs(increments[None, None] - diff[:, :, None]), dim = 2)
        approx_sum = a + values[closest]
        return approx_sum
    return add_func

def get_normalization_factor(precision, separation):

    # This function computes the average normalization factor
    # for the logarithmic Gaussian functions.

    ref_gauss_expon = -0.5
    other_gauss_expon = -0.5*(1 + separation**2 * precision)
    gauss_drift_term = ref_gauss_expon + other_gauss_expon
    log_sum_term = 0.5*separation**2 * precision
    normalization_factor = -(gauss_drift_term + log_sum_term)
    return normalization_factor

def get_analytic_tau(error_rate, gamma_m):

    # Thus function computes the optimal value of tau for the 
    # double-threshold algorithm.

    tau = -1.027e-6*np.log(9.6955*error_rate / gamma_m) / gamma_m
    return tau

# The following code runs numerical experiments using the optimal, linear,
# and logarithmic Bayesian models, as well as the double-threshold algorithm.

OPTIMAL = True
LINEAR = True
LOG = True
THRESH = True

T_list = [100] # Time of runs
gamma_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1] # Error rates
step_list = [100e-9] # Averaging interval

gamma_m = 2.5 # Rate baseline

thresh_fidels = []
results = {} # keys of form (duration, step, gamma)
np.random.seed(0)
torch.manual_seed(0)
for trial in range(10):
    for time_step in step_list:
        system = tools.Simulated(0.01*1e6, time_step, gamma_m*1e6, 0.5)
        for j in range(len(gamma_list)):
            gamma = gamma_list[j]
            system.set_gamma(gamma)
            for i in range(len(T_list)):
                T = T_list[i]

                (data_, initial, labels) = system.generate_exact_data(10000, T)
                data_ = data_ / time_step

                stop = 0
                (train_data, train_labels) = (data_[:stop], labels[:stop])
                (measurements, states) = (data_[stop:], labels[stop:])

                if OPTIMAL: optimal = Bayesian_Optimal(0)
                if LINEAR: bayesian_linear = Bayesian_Linear(0)
                if LOG: 
                    bayesian_log_max = Bayesian_Log(0, "max")
                    bayesian_log_two = Bayesian_Log(0, "half_exact")
                if THRESH: double_threshold = DoubleThreshold(get_analytic_tau(gamma, gamma_m), dt = time_step)

                if OPTIMAL: optimal.perfect_train_2(gamma_m, 0.5, gamma, time_step*1e6, 4000)
                if LINEAR: bayesian_linear.perfect_train(gamma_m, 0.5, gamma, time_step*1e6)
                if LOG: 
                    bayesian_log_max.perfect_train(gamma_m, 0.5, gamma, time_step*1e6)
                    bayesian_log_two.perfect_train(gamma_m, 0.5, gamma, time_step*1e6)

                if OPTIMAL:
                    optimal_acc = optimal.tracking_test(measurements.cuda(), states.cuda(), initial.cuda()) 
                    print("Optimal\n")
                if LINEAR: 
                    linear_acc = bayesian_linear.tracking_test(measurements, states, initial)
                    print("Linear\n")
                if LOG: 
                    log_max_acc = bayesian_log_max.tracking_test(measurements, states, initial)
                    print("Log Max\n")
                    log_two_acc = bayesian_log_two.tracking_test(measurements, states, initial)
                    print("Log Two\n")
                if THRESH:
                    stand_meas = measurements * time_step
                    stand_meas = stand_meas.cpu().numpy()
                    states = states.cpu().numpy()

                    R1, R2 = double_threshold.filter(stand_meas[:, :, 0], stand_meas[:, :, 1], states)

                    initial_states = initial
                    pred_states = double_threshold.identify_subspace(R1, R2, initial_states)
                    acc = np.sum(pred_states == states) / np.prod(pred_states.shape)
                    thresh_acc = np.sum(pred_states[:, -1] == states[:, -1]) / pred_states.shape[0]
                    print(f'Acc_{acc:.4f}')
                    print(f'Fidelity_{thresh_acc:.4f}')

                results[str((T, time_step, gamma))] = [optimal_acc, linear_acc, log_max_acc, log_two_acc, thresh_acc]   
