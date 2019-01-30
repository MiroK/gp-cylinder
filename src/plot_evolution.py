from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma 
import os, pickle

base = 'cxpb:0.6_mutpb:0.3_ngen:24_popsize:32'
gen = 15

# EVOLUTION OF MINIMUM -----------------------------------------------
best_history = 'best_history_%s.txt' % base
best = np.loadtxt(best_history)

fig, ax1 = plt.subplots()
ax1.plot(np.arange(1, len(best)+1), best, 'bx--')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Fitness')
ax1.tick_params('y', color='b')

# Againt nevels
nevals = 'nevals_history_%s.txt' % base
nevals = np.cumsum(np.loadtxt(nevals))

ax2 = ax1.twinx()
ax2.plot(np.arange(1, len(nevals)+1), nevals, 'gx:')
ax2.set_ylabel('# problem solves')
ax2.tick_params('x', colors='g')

# EVOLUTION OF POPULATOIN --------------------------------------------
pop_history = 'pop_history_%s.txt' % base
pop = np.loadtxt(pop_history)

# Sort each generation
pop = np.sort(pop, axis=1)
pop = pop.T # to have generation with x

pop = ma.masked_greater(pop, 1E1)

fig = plt.figure()
ax = fig.gca()
mappable = ax.pcolor(pop,
                     # norm=LogNorm(vmin=pop.min(), vmax=pop.max()),
                     cmap='Spectral')
                     #edgecolors='yellow', linewidths=0.1)

contour = ax.contour(pop,
                     levels=[5, 1, 0.5, 0.25],
                     # norm=LogNorm(vmin=pop.min(), vmax=pop.max()),
                     cmap='Spectral')
                     #colors='black')

ax.clabel(contour, fmt='%.2E', colors='black', fontsize=14)

cbar = fig.colorbar(mappable, ax=ax)
cbar.set_label('Fitness')

plt.yticks([])

plt.xlabel('Generation')
plt.xticks(np.arange(1, pop.shape[1]+1)-0.5)
ax.set_xticklabels(np.arange(1, pop.shape[1]+1))

plt.show()
