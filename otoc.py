# This script evaluates out-of-time ordered correlator (OTOC) shown in Fig. 2(c) in the article.

from functools import partial
from pathlib import Path

import numpy as np
from qutip import sigmaz
from qutip_qip.operations import expand_operator
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import reservoir_cf, position, hamiltonian, timing
from utils import mpl_rcParams, figwidth_sc
mpl.rcParams.update(mpl_rcParams)


def otoc(H, tlist):
    '''OTOC at infinite temperature'''
    n_qubits = len(H.dims[0])
    Z = [expand_operator(sigmaz(), dims=H.dims[0], targets=i) for i in range(n_qubits)]

    OTOC = np.zeros(len(tlist))
    for i in range(len(tlist)):
        U = (-1j*H*tlist[i]).expm()
        Z0 = U.dag() * Z[0] * U
        OTOC[i] = np.mean([(Z0*Z[j]*Z0*Z[j]).tr().real/H.shape[0] for j in range(1, n_qubits)])
    return OTOC

def worker(n_qubits, tlist, seed):
    pos = position(n_qubits, reservoir_cf.a, reservoir_cf.lattice, sigma=reservoir_cf.sigma, seed=seed)
    H = hamiltonian(reservoir_cf.Delta0, pos)
    return otoc(H, tlist)


@timing
def main(plot=False):
    n_qubits = 10
    tlist = np.logspace(0, 2, 21)
    n_samples = 10

    # run in parallel
    result = list(process_map(partial(worker, n_qubits, tlist), np.arange(n_samples), max_workers=n_samples))
    OTOC = np.array(result) # OTOC is of shape (n_samples, len(tlist))

    # save data
    path = Path('data/otoc')
    path.mkdir(parents=True, exist_ok=True)
    np.save(path/'OTOC.npy', OTOC)

    # plot results
    if plot:
        fig, ax = plt.subplots(layout='constrained', figsize=(figwidth_sc, figwidth_sc*.8))
        mean = np.mean(OTOC, axis=0)
        std = np.std(OTOC, axis=0)
        ax.semilogx(tlist, mean)
        ax.fill_between(tlist, mean+std, mean-std, alpha=.5)
        ax.set_xlabel(r'$\Omega\tau$')
        ax.set_ylabel(r'$F(\tau)$')
        fig.savefig('figures/otoc.pdf', bbox_inches='tight')

if __name__=='__main__':
    main(plot=True)