import numpy as np
from scipy import random
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal

def gen_data(k=3, dim=2, points_per_cluster=200, lim=[-10, 10]):
    '''
    Generates data from a random mixture of Gaussians in a given range.
    input:
        - k: Number of Gaussian clusters
        - dim: Dimension of generated points
        - points_per_cluster: Number of points to be generated for each cluster
        - lim: Range of mean values
    output:
        - X: Generated points (points_per_cluster*k, dim)
    '''
    x = []
    mean = random.rand(k, dim)*(lim[1]-lim[0]) + lim[0]
    for i in range(k):
        cov = random.rand(dim, dim+10)
        cov = np.matmul(cov, cov.T)
        _x = np.random.multivariate_normal(mean[i], cov, points_per_cluster)
        x += list(_x)
    x = np.array(x)
    return x

class Gauss_mix_model():
    def __init__(self, X, k = 2, dim = 1):
        '''
        Parameters:
        k: Number of Gaussian components/ clusters
        dim: Dimension of each Gaussian components
        mu: Mean value of each Gaussian component (k,dim)
            - initial values random from uniform[-10, 10]
        sigma: diagonal covariance matrix for each component (k, dim, dim)
        pi: component/ cluster weights
            - initial value is equal for all clusters = 1/k
        '''
        self.k = k
        self.dim = dim
        # Create initialization value for the parameters
        self.mu= random.rand(k, dim)*20 - 10
        init_sigma = np.zeros((k, dim, dim))
        for i in range(k):
            init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        self.pi = np.ones(self.k)/self.k

        self.data = X
        self.n_pts = X.shape[0] #no. of datapoints
        self.gamma = np.zeros((self.n_pts, self.k)) #responsibilities container

    def e_step(self):
        '''
        Evaluate responsibilities given current model parameters and data
        '''
        for i in range(self.k):
            self.gamma[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        self.gamma /= self.gamma.sum(axis=1, keepdims=True)

    def m_step(self):
        '''
        Re-estimate mixture distribution parameters (pi, mu, sigma) given current responsibilities
        '''
        sum_gamma = self.gamma.sum(axis=0)
        self.pi = sum_gamma / self.n_pts # recalculate mixture weights
        self.mu = np.matmul(self.gamma.T, self.data)
        self.mu /= sum_gamma[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.gamma[:, i] )
            self.sigma[i] /= sum_gamma[i]

    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            ll.append(np.log(tot))
        return np.sum(ll)

if __name__ == "__main__":
    # Generate random 2D data with 3 clusters
    X = gen_data(k=3, dim=2, points_per_cluster=1000)
    # Instantiating the Gaussian Mixture Model class
    gmm = Gauss_mix_model(X,3,2)
    num_iters = 30 # no. of training iteration
    log_likelihood = [gmm.log_likelihood(X)] #storing the result
    for e in range(num_iters):
        gmm.e_step() # E-step
        gmm.m_step() # M-step
        log_likelihood.append(gmm.log_likelihood(X)) # Calculate log-likelihood
        print("Iteration: {}, log-likelihood: {:.4f}".format(e+1, log_likelihood[-1])) #Display
