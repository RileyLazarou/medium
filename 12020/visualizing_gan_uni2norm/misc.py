import os

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def uniform_to_normal(z):
    '''
    Map a value from ~U(-1, 1) to ~N(0, 1)
    '''
    norm = stats.norm(0, 1)
    return norm.ppf((z+1)/2)

grid_latent = np.linspace(-1, 1, 203)[1:-1].reshape((-1, 1))
true_mappings = uniform_to_normal(grid_latent)

chunky = []
t = []
for i in range(len(grid_latent.flatten())):
    t.append((grid_latent[i, 0] + 1) / 2)
    if np.random.rand() < 0.1:
        if np.random.rand() < 0.5:
            t = t[::-1]
        chunky.append(t)
        t = []
else:
    chunky.append(t)
np.random.shuffle(chunky)
piecewise = []
for i in chunky:
    piecewise += i
piecewise = np.array(piecewise)



plt.figure(figsize=(6,6))
plt.scatter(grid_latent.flatten(), ((grid_latent.flatten() + 1) / 2), 
            edgecolor='blue', facecolor='None', s=5, alpha=1, 
            linewidth=1, label='f(z)=(z+1)/2')
plt.scatter(grid_latent.flatten(), ((1 - grid_latent.flatten()) / 2), 
            edgecolor='green', facecolor='None', s=5, alpha=1, 
            linewidth=1, label='f(z)=(1-z)/2')
plt.scatter(grid_latent.flatten(), piecewise, 
            edgecolor='red', facecolor='None', s=5, alpha=1, 
            linewidth=1, label='f(z)=[piecewise function]')
plt.xlim(-1, 1)
plt.ylim(0, 1)
plt.xlabel('Latent Space (z)')
plt.ylabel('Transformed Space (f(z))')
plt.legend(loc=8)
plt.tight_layout()
plt.savefig(f'piecewise_f.png')
plt.close()



plt.figure(figsize=(6,6))
plt.scatter(grid_latent.flatten(), true_mappings.flatten(), 
            edgecolor='blue', facecolor='None', s=5, alpha=1, 
            linewidth=1, label='f(z)=(z+1)/2')
plt.scatter(grid_latent.flatten(), true_mappings.flatten()[::-1], 
            edgecolor='green', facecolor='None', s=5, alpha=1, 
            linewidth=1, label='f(z)=(1-z)/2')
plt.scatter(grid_latent.flatten(), stats.norm(0, 1).ppf(piecewise), 
            edgecolor='red', facecolor='None', s=5, alpha=1, 
            linewidth=1, label='f(z)=[piecewise function]')
plt.xlim(-1, 1)
plt.ylim(-3, 3)
plt.xlabel('Latent Space (z)')
plt.ylabel('Sample Space (x)')
plt.legend(loc=8)
plt.tight_layout()
plt.savefig(f'piecewise_g.png')
plt.close()
