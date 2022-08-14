import numpy as np
from scipy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
from sklearn.preprocessing import normalize
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal

# Generate synthetic data for density estimation
def gen_data(k=3, dim=1, points_per_cluster=5000, lim=[-10, 10]):
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

if __name__ == "__main__":
    samp, mu, co  = gen_data() #obtain samples

    #Set parameters
    M = 10 #no. of bins
    b_len = abs(np.max(samp)-np.min(samp))/M # get bin length
    bins = np.arange(np.min(samp),np.max(samp), b_len) #get bin levels

    # Calculate binning counts
    ar = np.zeros((M,1))
    for j in samp:
        getind = (np.where(bins<=j)[0])
        ind = getind[-1]
        ar[ind] += 1
    ar = normalize(ar, norm='max', axis =0)
    # Plot histogram
    plt.plot()
    plt.bar(bins.tolist(),np.squeeze(ar.tolist()), width = 1.5, align='center', color = 'maroon', label = 'Histogram')

    #Plot the data generating distribution
    cols = ['blue','forestgreen','orangered']
    lab = ['Dist01','Dist02','Dist03']
    for t in range(3):
        plt.plot(bins, norm.pdf(bins, mu[t], co[t]),color = cols[t], linewidth = 2, label = lab[t])
    plt.legend()
    plt.show()
