import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Observed data: H=Head, T=Tail
data = ['H', 'H', 'T', 'T', 'H', 'H', 'H']

# Prior parameters
a_prior = 5
b_prior = 5

# Count the number of heads and tails
n_heads = data.count('H')
n_tails = data.count('T')

# Update parameters of the posterior distribution
a_posterior = a_prior + n_heads
b_posterior = b_prior + n_tails

# Generate values for theta
x_theta = np.linspace(0, 1, 1000)

# Compute the posterior distribution
y_posterior = beta.pdf(x_theta, a_posterior, b_posterior)

# Plot the likelihood function and the posterior distribution
plt.plot(x_theta, y_posterior, label='beta-Likelihood', color='blue')
plt.xlabel('Theta')
plt.ylabel('Density')
plt.title('beta-Likelihood Distribution')
plt.legend()
plt.grid(True)
plt.show()

# Find the MAP estimate
theta_map = theta_values[np.argmax(posterior)]
print("MAP Estimate of Theta:", theta_map)
