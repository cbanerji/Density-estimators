# generate synthetic dataset
import numpy as np
from scipy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import math
import itertools
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal

# Generate synthetic data for density estimation
def gen_data(k=3, dim=1, points_per_cluster=100, lim=[-10, 10]):
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
    cove = []
    mean = random.rand(k, dim)*(lim[1]-lim[0]) + lim[0]
    for i in range(k):
        cov = random.rand(dim, dim+10)
        cov = np.matmul(cov, cov.T)
        _x = np.random.multivariate_normal(mean[i], cov, points_per_cluster)
        x += list(_x)
        cove.append(cov)
    x = np.clip(np.array(x),-10,10)
    return x, np.squeeze(mean), np.squeeze(cove)

# Define Gaussian function
def gauss(j_pass, mu=0.0, sigma=1.0):
   '''
   Return the value of the Gaussian probability function with
   mean (mu) and standard deviation (sigma) at the given x value.
   '''
   j_pass = float(j_pass - mu) / sigma
   return math.exp(-j_pass*j_pass/2.0) / math.sqrt(2.0*math.pi) / sigma

def cal_den(x, sigma):
    den = []
    n = x.size
    for j in x:
        sm = []
        for m in x:
            j_pass = (j-m)/sigma
            gauss_val = gauss(j_pass, sigma)
            sm.append(gauss_val)
        sm_sum = sum(sm)
        kde = sm_sum/ (n*sigma)
        den.append(kde)
    fin_den.append(den)
    return fin_den

if __name__ == "__main__":
    fin_den = []
    sigma  = 3
    x,mu,co = gen_data()
    dnsi = cal_den(x, sigma)
    dnsi = [dnsi for sublist in dnsi for dnsi in sublist] #Flatten
    #plot
    a_val = np.linspace(np.min(dnsi),np.max(dnsi), len(dnsi))
    val = np.arange(np.min(x),np.max(x), 0.5)
    for t in range(3):
        plt.plot(val, norm.pdf(val, mu[t], co[t]))
    plt.plot(a_val, dnsi)
    plt.show()
