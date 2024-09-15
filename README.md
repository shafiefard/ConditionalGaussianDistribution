There are two examples of the conditional Gaussian distribution with Python (Jupyter Notebook) code, followed by a step-by-step explanation:(conditionalGaussianDistribution.ipynb)
Step 1: Data Generation

The first step is generating multivariate data that follows a specific relationship between the variables. We create four variables, where two are linearly dependent on others, introducing correlations into the dataset.

python

import numpy as np
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(37)

# Generate sample data
N = 1000
x1 = np.random.normal(1, 1, N)  # x1: normal distribution with mean 1 and std dev 1
x2 = np.random.normal(1 + 3.5 * x1, 1, N)  # x2 depends linearly on x1
x3 = np.random.normal(2, 1, N)  # x3: normal distribution with mean 2
x4 = np.random.normal(3.8 - 2.5 * x3, 1, N)  # x4 depends linearly on x3

# Combine data into a single array
data = np.vstack([x1, x2, x3, x4]).T

Explanation:

    We generate four variables (x1, x2, x3, x4), where x2 is dependent on x1, and x4 is dependent on x3. This creates correlations between the variables.
    The final dataset data is an N×4N×4 matrix, where N=1000N=1000.

Step 2: Calculate Basic Statistical Properties

Next, we calculate basic statistics such as the mean, covariance matrix, standard deviations, and the correlation matrix for the dataset.

python

# Calculate statistical properties of the dataset
means = data.mean(axis=0)  # Mean of each variable
cov = np.cov(data.T)  # Covariance matrix
std = np.sqrt(np.diag(cov))  # Standard deviations of the variables
cor = np.corrcoef(data.T)  # Correlation matrix

# Display key statistical properties
print('Means:', means)
print('Covariance Matrix:\n', cov)
print('Standard Deviations:', std)
print('Correlation Matrix:\n', cor)

Explanation:

    means: Calculates the mean for each variable in the dataset.
    cov: Computes the covariance matrix, which shows the relationships between the variables.
    std: Extracts the standard deviations from the diagonal of the covariance matrix.
    cor: Computes the correlation matrix, showing the strength and direction of linear relationships between variables.

Step 3: Partition the Data

To model the conditional distribution P(xa∣xb)P(xa​∣xb​), we need to partition the dataset, means, and covariance matrix into two subsets.

    Partition Means: Split the mean vector into two subsets for xaxa​ and xbxb​.
    Partition Covariance Matrix: Split the covariance matrix into four submatrices corresponding to the covariances and cross-covariances between xaxa​ and xbxb​.
    Partition Data Vector: Similarly, split each data point into two parts.

python

# Function to partition means into two subsets
def partition_means(index_1, means, index_2=None):
    index_2 = [i for i in range(len(means)) if i not in index_1] if index_2 is None else index_2
    m_1, m_2 = means[index_1], means[index_2]
    return m_1, m_2

# Function to partition covariance matrix into blocks
def partition_cov(index_1, cov, index_2=None):
    index_2 = [i for i in range(cov.shape[1]) if i not in index_1] if index_2 is None else index_2
    s_11 = cov[index_1][:, index_1]  # Covariance of subset 1
    s_12 = cov[index_1][:[:, index_2]  # Cross-covariance between subset 1 and 2
    s_21 = cov[index_2][:, index_1]  # Cross-covariance between subset 2 and 1
    s_22 = cov[index_2][:, index_2]  # Covariance of subset 2
    return s_11, s_12, s_21, np.linalg.inv(s_22)

# Function to partition data vector x into two subsets
def partition_x(index_1, x, index_2=None):
    index_2 = [i for i in range(len(x)) if i not in index_1] if index_2 is None else index_2
    x_1 = x[index_1]
    x_2 = x[index_2]
    return x_1, x_2

Explanation:

    These functions split the means, covariance, and data vectors into subsets based on index_1 (for xaxa​) and index_2 (for xbxb​).

Step 4: Compute the Conditional Distribution P(xa∣xb)P(xa​∣xb​)

Finally, we compute the log-probability for the conditional distribution based on the formula:
P(xa∣xb)∼N(μb+ΣbaΣaa−1(xa−μa),Σbb−ΣbaΣaa−1Σab)
P(xa​∣xb​)∼N(μb​+Σba​Σaa−1​(xa​−μa​),Σbb​−Σba​Σaa−1​Σab​)

python

# Function to compute the log probability for the conditional distribution
def get_log_proba(index_1, data, means, cov, index_2=None, zero=1e-6):
    """
    Calculates the log-probability of x_a given x_b based on the conditional 
    multivariate normal distribution: P(x_a | x_b).
    
    :param index_1: Indices corresponding to x_a (subset 1)
    :param data: Dataset (N samples x M variables)
    :param means: Mean vector of the full dataset
    :param cov: Covariance matrix of the full dataset
    :param index_2: Optional indices corresponding to x_b (subset 2)
    :param zero: Small threshold for log computation to avoid log(0)
    :return: Sum of log-probabilities for all data points and the regression coefficients
    """
    # Partition means and covariance matrix
    m_1, m_2 = partition_means(index_1, means, index_2)
    s_11, s_12, s_21, s_22 = partition_cov(index_1, cov, index_2)
    
    # Conditional variance and inverse covariance
    conditional_variance = (s_11 - s_12.dot(s_22).dot(s_21))[0, 0]

    log_proba = []
    for x in data:
        # Partition the data vector into x_a and x_b
        x_1, x_2 = partition_x(index_1, x, index_2)
        
        # Conditional mean of x_a given x_b
        conditional_mean = (m_1 + s_12.dot(s_22).dot((x_2 - m_2).T))[0]
        
        # Compute normal PDF for x_a given the conditional distribution
        p = norm.pdf(x_1, loc=conditional_mean, scale=np.sqrt(conditional_variance))
        log_p = np.log(p) if p >= zero else 0.0  # Avoid log(0)
        log_proba.append(log_p)

    return sum(log_proba), s_12.dot(s_22)[0]

# Example usage for conditional probability: P(x2 | x1)
index_1 = [1]  # Index of x2
log_probability, regression_coefficients = get_log_proba(index_1, data, means, cov)

print('Log-probability:', log_probability)
print('Regression Coefficients:', regression_coefficients)

Explanation:

    Log-Probability: The get_log_proba function calculates the log-probability for each data point based on the conditional distribution.

    Conditional Mean: The formula for the conditional mean is:
    μcond=μb+ΣbaΣaa−1(xa−μa)
    μcond​=μb​+Σba​Σaa−1​(xa​−μa​)

    Conditional Variance: The conditional variance is:
    Σcond=Σbb−ΣbaΣaa−1Σab
    Σcond​=Σbb​−Σba​Σaa−1​Σab​

    The function computes the log-probabilities and returns the sum of log-probabilities across all samples, along with the regression coefficients.
