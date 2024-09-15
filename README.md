this is an example of conditional Gaussian distribution with python (jupyter notebook) code
Step-by-Step Code Explanation:
1. Import Libraries

We need numpy for handling arrays and basic linear algebra, and multivariate_normal from scipy.stats for sampling from a multivariate normal distribution.

python

import numpy as np
from scipy.stats import multivariate_normal

2. Set Seed for Reproducibility

To ensure that the random values generated are the same each time you run the code, we set a seed.

python

np.random.seed(37)

3. Generate Data

We generate three random variables x1x1​, x2x2​, and x3x3​ with dependencies between them.

    x1∼N(0,1)x1​∼N(0,1) (independent)
    x2∼N(2+1.5⋅x1,1)x2​∼N(2+1.5⋅x1​,1) (dependent on x1x1​)
    x3∼N(−1+0.8⋅x2,1)x3​∼N(−1+0.8⋅x2​,1) (dependent on x2x2​)

This sets up a simple example where one variable depends on the others.

python

# Number of samples
N = 1000

# Generate random data
x1 = np.random.normal(0, 1, N)  # x1 ~ N(0, 1)
x2 = np.random.normal(2 + 1.5 * x1, 1, N)  # x2 ~ N(2 + 1.5 * x1, 1)
x3 = np.random.normal(-1 + 0.8 * x2, 1, N)  # x3 ~ N(-1 + 0.8 * x2, 1)

4. Stack Data

We create a data matrix where each row contains a sample of [x1,x2,x3][x1​,x2​,x3​].

python

# Stack data into a matrix
data = np.vstack([x1, x2, x3]).T

5. Compute Statistics

We compute the means and the covariance matrix of the dataset, which will be used in the conditional distribution formula.

    means: The vector μμ of the means of x1x1​, x2x2​, and x3x3​.
    cov: The covariance matrix ΣΣ of the variables.

python

# Compute statistics
means = data.mean(axis=0)
cov = np.cov(data.T)

6. Partition Means and Covariance Matrix

We define a helper function that partitions the mean vector and the covariance matrix based on a subset of indices.

    μaμa​: Mean vector of the variables xaxa​.
    μbμb​: Mean vector of the remaining variables xbxb​.
    ΣaaΣaa​, ΣabΣab​, ΣbaΣba​, ΣbbΣbb​: Partitioned covariance matrices as per the formula.

python

def partition_means_cov(index_a, means, cov):
    index_b = [i for i in range(len(means)) if i not in index_a]
    
    mu_a, mu_b = means[index_a], means[index_b]
    Sigma_aa = cov[np.ix_(index_a, index_a)]
    Sigma_ab = cov[np.ix_(index_a, index_b)]
    Sigma_ba = cov[np.ix_(index_b, index_a)]
    Sigma_bb = cov[np.ix_(index_b, index_b)]
    
    return mu_a, mu_b, Sigma_aa, Sigma_ab, Sigma_ba, Sigma_bb

7. Conditional Distribution Calculation

We define a function that calculates the conditional distribution of \mathbitxb\mathbitxb​ given \mathbitxa\mathbitxa​ based on the formula:

    Conditional Mean:
    μb+ΣbaΣaa−1(\mathbitxa−μa)
    μb​+Σba​Σaa−1​(\mathbitxa​−μa​)

    Conditional Covariance:
    Σbb−ΣbaΣaa−1Σab
    Σbb​−Σba​Σaa−1​Σab​

This function uses the indices of \mathbitxa\mathbitxa​ to compute the conditional mean and covariance for \mathbitxb\mathbitxb​.

python

def conditional_distribution(index_a, x_a, means, cov):
    mu_a, mu_b, Sigma_aa, Sigma_ab, Sigma_ba, Sigma_bb = partition_means_cov(index_a, means, cov)
    
    # Compute the inverse of Sigma_aa
    Sigma_aa_inv = np.linalg.inv(Sigma_aa)
    
    # Conditional mean: μ_b + Σ_ba Σ_aa^(-1) (x_a - μ_a)
    conditional_mean = mu_b + Sigma_ba.dot(Sigma_aa_inv).dot(x_a - mu_a)
    
    # Conditional covariance: Σ_bb - Σ_ba Σ_aa^(-1) Σ_ab
    conditional_cov = Sigma_bb - Sigma_ba.dot(Sigma_aa_inv).dot(Sigma_ab)
    
    return conditional_mean, conditional_cov

8. Choose Variables to Condition On

We now specify which variables to condition on. In this case, we are conditioning on x1x1​ (i.e., \mathbitxa=x1\mathbitxa​=x1​) and predicting x2x2​ and x3x3​ (i.e., \mathbitxb=[x2,x3]\mathbitxb​=[x2​,x3​]).

python

# Select index_a as the first variable (x1)
index_a = [0]

# Choose a value of x_a to condition on (mean of x1 in this case)
x_a = np.array([means[0]])

9. Compute Conditional Mean and Covariance

We compute the conditional mean and covariance of x2x2​ and x3x3​ given the value of x1x1​.

python

# Compute the conditional distribution of x_b given x_a
conditional_mean, conditional_cov = conditional_distribution(index_a, x_a, means, cov)

10. Display Results

Print the computed conditional mean and covariance for x2x2​ and x3x3​, given the value of x1x1​.

python

# Show the results
print("Conditional mean of x_b given x_a:")
print(conditional_mean)
print("Conditional covariance of x_b given x_a:")
print(conditional_cov)

11. Draw Samples from the Conditional Distribution

Finally, we can sample from the computed conditional distribution to generate values of x2x2​ and x3x3​, given x1x1​.

python

# Now, draw samples from the conditional distribution of x_b
conditional_samples = multivariate_normal.rvs(mean=conditional_mean, cov=conditional_cov, size=N)

# Show some sample conditional values
print("\nConditional samples (first 5 rows):")
print(conditional_samples[:5])
