# This script calculates the phase diagram of the atom array utilizing the level spacing ratio, shown in Fig. 2(b) in the article.

from pathlib import Path
from itertools import product
from functools import partial

import numpy as np
from mpi4py.futures import MPIPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl

from utils import reservoir_cf, position, hamiltonian, timing
from utils import mpl_rcParams, figwidth_sc
mpl.rcParams.update(mpl_rcParams)

def mean_level_spacing(E) -> float:
    delta = E[1:] - E[:-1]
    r = np.mean([min(delta[i], delta[i+1]) / max(delta[i], delta[i+1]) for i in range(len(delta)-1)])
    return r

def phase_diagram(n_qubits, a, Delta, seed) -> np.ndarray:
    '''Compute <r> with respect to lattice constant a and detuning Delta'''
    r = []
    params = product(a, Delta)
    for _a, _Delta in params:
        pos = position(n_qubits, _a, reservoir_cf.lattice, reservoir_cf.sigma, seed)
        H = hamiltonian(_Delta, pos)
        E = H.eigenenergies()
        r.append(mean_level_spacing(E))
    r = np.array(r).reshape((len(a), len(Delta)))
    return r

@timing
def main(plot=False):
    n_qubits = 10
    a = np.linspace(.7, 1.6, 201)
    Delta = np.linspace(-9, 3, 201)
    n_samples = 1000

    # run in parallel
    with MPIPoolExecutor() as executor:
        result = list(executor.map(partial(phase_diagram, n_qubits, a, Delta), np.arange(n_samples)))
        r = np.array(result) # r is of shape (n_samples, len(a), len(Delta))
    
    # save the data
    path = Path('data/phase')
    path.mkdir(parents=True, exist_ok=True)
    np.save(path/'r.npy', r)

    # plot the results
    if plot:
        fig, ax = plt.subplots(layout='constrained', figsize=(figwidth_sc, figwidth_sc*.8))
        r_mean = np.mean(r, axis=0) # average over atomic positions
        Delta, a = np.meshgrid(Delta, a)
        im = ax.pcolor(Delta, a, r_mean, cmap='bwr', vmin=0.38629, vmax=0.53590)
        cbar = fig.colorbar(im, ax=ax, location='right', ticks=MultipleLocator(.03), pad=0)
        cbar.set_label(label=r'$\langle r\rangle$', labelpad=-.5)
        cbar.ax.tick_params(which='major', direction='in', length=1, pad=.5)
        ax.scatter(-.5, 1, s=4**2, c='k', marker='x')
        ax.xaxis.set_major_locator(MultipleLocator(3))
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.set_xlabel('$\Delta/\Omega$')
        ax.set_ylabel('$a/R_b$')
        fig.savefig('figures/phase.pdf', bbox_inches='tight')

if __name__=='__main__':
    main(plot=True)