import numpy as np
from scipy import special as sci

# This module contains functions that can be used to compute
# a variety of syndrome densities under different conditions.

def mean_density(n, x):

    # This functions computes the probability density from the 
    # beta distribution describing a single syndome given n errors.

    neg_power = (n - 1) // 2
    pos_power = (n - 1) - neg_power
    unnorm = (1 + x)**pos_power * (1 - x)**neg_power
    norm = np.math.factorial(n)/(2**n * np.math.factorial(neg_power) * np.math.factorial(pos_power))
    y = unnorm * norm
    return y

def marg_mean_density(x, p_avg, T):
    
    # This function computes the syndrome density after 
    # marginalizing over all possible error numbers.

    hyp_arg = (p_avg/2)**2 * (1 - (x/T)**2)
    hyp_1 = sci.hyp0f1(1, hyp_arg)
    hyp_2 = sci.hyp0f1(2, hyp_arg)
    density = p_avg*np.exp(-p_avg)/2 * (hyp_1 + p_avg*(1 + x/T)*hyp_2/2) / T
    return density

def get_partial_means(num_samples, error_triplet):

    # This function samples syndrome means for a number of errors
    # across the three qubits.

    (num_1, num_2, num_3) = error_triplet
    positions = np.random.uniform(size = [num_samples, num_1 + num_2 + num_3])
    synd_1 = np.concatenate([positions[:, :num_1], positions[:, num_1:num_1 + num_2]], axis = 1)
    synd_2 = np.concatenate([positions[:, num_1:num_1 + num_2], positions[:, num_1 + num_2:]], axis = 1)
    partial_means_1 = integrate_synds(synd_1)
    partial_means_2 = integrate_synds(synd_2)
    partial_means = np.stack([partial_means_1, partial_means_2], axis = 1)
    return partial_means

def integrate_synds(synds):

    # This function computes the mean syndrome value.

    num_samples = synds.shape[0]
    synd_sorted = np.sort(synds, axis = 1)
    points = np.concatenate([np.zeros([num_samples, 1]), synd_sorted, np.ones([num_samples, 1])], axis = 1)
    segments = points[:, 1:] - points[:, :-1]
    segments[:, 1::2] = -segments[:, 1::2]
    partial_means = segments.sum(1)
    return partial_means

def get_signal_samples(num_samples, error_triplet, variance):

    # This function samples measurement readouts by adding Gaussian
    # noise to the syndrome means.

    means = get_partial_means(num_samples, error_triplet)
    signal_samples = np.random.normal(means, variance)
    return signal_samples

def flip_2_density(X, Y, variance):

    # This function computes the syndrome densities for an error
    # on the second qubit.

    sum_variance = variance / 2
    diff_variance = (1 + 3*sum_variance) / 3
    expon_sum = -0.5*((X - Y)/2)**2 / sum_variance + 0.5*np.math.log(1/(2*np.math.pi*sum_variance))
    expon_diff = -0.5*((X + Y)/2)**2 / diff_variance + 0.5*np.math.log(1/(2*np.math.pi*diff_variance))
    density = np.exp(expon_sum + expon_diff - np.math.log(2))
    return density
