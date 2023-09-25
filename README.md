# Bayesian Inference for Quantum Error Correction
Implements in PyTorch the Bayesian error correction models described in [A Logarithmic Bayesian Approach to Quantum Error Detection (2022)](https://arxiv.org/abs/2110.10732):
> We consider the problem of continuous quantum error correction from a Bayesian perspective, proposing a pair of digital filters using logarithmic probabilities that are able to achieve near-optimal performance on a three-qubit bit-flip code, while still being reasonable to implement on low-latency hardware. These practical filters are approximations of an optimal filter that we derive explicitly for finite time steps, in contrast with previous work that has relied on stochastic differential equations such as the Wonham filter. By utilizing logarithmic probabilities, we are able to eliminate the need for explicit normalization and can reduce the Gaussian noise distribution to a simple quadratic expression. The state transitions induced by the bit-flip errors are modeled using a Markov chain, which for log-probabilties must be evaluated using a LogSumExp function. We develop the two versions of our filter by constraining this LogSumExp to have either one or two inputs, which favors either simplicity or accuracy, respectively. Using simulated data, we demonstrate that the single-term and two-term filters are able to significantly outperform both a double threshold scheme and a linearized version of the Wonham filter in tests of error detection under a wide variety of error rates and time steps.

and [Machine Learning for Continuous Quantum Error Correction on Superconducting Qubits (2022)](https://arxiv.org/abs/2110.10378):

> Continuous quantum error correction has been found to have certain advantages over discrete quantum error correction, such as a reduction in hardware resources and the elimination of error mechanisms introduced by having entangling gates and ancilla qubits. We propose a machine learning algorithm for continuous quantum error correction that is based on the use of a recurrent neural network to identify bit-flip errors from continuous noisy syndrome measurements. The algorithm is designed to operate on measurement signals deviating from the ideal behavior in which the mean value corresponds to a code syndrome value and the measurement has white noise. We analyze continuous measurements taken from a superconducting architecture using three transmon qubits to identify three significant practical examples of non-ideal behavior, namely auto-correlation at temporal short lags, transient syndrome dynamics after each bit-flip, and drift in the steady-state syndrome values over the course of many experiments. Based on these real-world imperfections, we generate synthetic measurement signals from which to train the recurrent neural network, and then test its proficiency when implementing active error correction, comparing this with a traditional double threshold scheme and a discrete Bayesian classifier. The results show that our machine learning protocol is able to outperform the double threshold protocol across all tests, achieving a final state fidelity comparable to the discrete Bayesian classifier.

whose abstracts have been reproduced above.

The `examples.ipynb` Jupyter notebook provides an overview of the quantum measurement data, and a simple version of the Bayesian algorithm used to filter it. The full model code can be found within the `log_bayes.py` and `real_data.py` modules in the `src` directory.

A non-interactive write-up can be found on [my website](https://ianconvy.github.io/projects/phd/bayesian/bayesian.html).
