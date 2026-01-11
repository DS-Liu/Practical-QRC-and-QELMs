# This script evaluates the IPC with respect to the decay rate. The results are shown in Fig. 3 (b) and (d)

import pickle
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

import numpy as np
from qutip import tensor, fock_dm, mesolve
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator

from detuning import qelm_evolve, calculate_ipc
from utils import reservoir_cf, hamiltonian, position, expect_ops, collapse_ops, timing
from utils import mpl_rcParams, figwidth_sc
mpl.rcParams.update(mpl_rcParams)

def qrc_evolve(seed, Delta0, decay, verbose=False):
    n_qubits = 10
    u = np.load('data/u.npy')
    u = np.hstack((u, np.array([0.])))
    tlist = reservoir_cf.tau * np.arange(len(u))
    H = hamiltonian(Delta0 + .1*u, position(n_qubits, reservoir_cf.a, reservoir_cf.lattice, sigma=reservoir_cf.sigma, seed=seed), tlist)
    rho0 = tensor([fock_dm(2, 1)]*n_qubits)
    e_ops = expect_ops(n_qubits, k=2)
    c_ops = collapse_ops(n_qubits, decay)

    progress_bar = 'tqdm' if verbose else ''
    result = mesolve(H, rho0, tlist, c_ops, e_ops, options=dict(nsteps=int(1e8), progress_bar=progress_bar))
    X = np.array(result.expect).T
    return X

@timing
def main(args):
    path = Path('data/decay')
    path.mkdir(parents=True, exist_ok=True)
    u = np.load('data/u.npy')
    decay = np.logspace(-1, 2, num=16)

    if args.qrc_evolve:
        with MPIPoolExecutor() as executor:
            for seed in range(5):
                verbose = [False]*(len(decay)-1) + [True]
                X = list(executor.map(partial(qrc_evolve, seed, reservoir_cf.Delta0), decay, verbose))
                np.save(path/f'X_qrc_seed_{seed}.npy', X)
    
    if args.qelm_evolve:
        with MPIPoolExecutor() as executor:
            n_qubits = 10
            u_pad = np.hstack(([0]*(n_qubits-1), u))
            
            for seed in range(5):
                X = []
                for d in tqdm(decay):
                    Delta = reservoir_cf.Delta0 + .1*np.array([u_pad[i:i+n_qubits] for i in range(len(u))])
                    X.append(list(executor.map(partial(qelm_evolve, seed, d), Delta)))
                    np.save(path/f'X_qelm_seed_{seed}.npy', X)

    if args.calculate_ipc:
        for seed in range(5):
            X = np.load(path/'X_qrc_seed_{seed}.npy')
            calculate_ipc(u, X, dir=(path/'ipc_qrc_seed_{seed}').as_posix())

            X = np.load(path/f'X_qelm_seed_{seed}.npy')
            calculate_ipc(u, X, dir=(path/f'ipc_qelm_seed_{seed}').as_posix())

    if args.plot_ipc:
        fig, axd = plt.subplot_mosaic([['(b)'], ['(d)']], layout='constrained', figsize=(figwidth_sc*.5, figwidth_sc*.75))
        r = np.mean(np.load('data/phase/r.npy'), axis=0)
        th_scale = 1.395

        # ipc of qrc with respect to decay rate
        ipcs_degree = []
        for seed in range(5):
            with open(f'data/decay/ipc_qrc_seed_{seed}/ipc_result.pickle', 'rb') as f:
                ipc_result = pickle.load(f)
            ipc, pkldir, paths = ipc_result['ipc'], ipc_result['pkldir'], ipc_result['paths']
            npzname = f'{pkldir}/state_indicators.npz'
            ipc.get_indicators(npzname, paths, th_scale=th_scale)
            assert np.all(ipc.Ctots<ipc.ranks)
            ipcs_degree.append(ipc.ipcs_degree)
        ipcs_degree = np.mean(ipcs_degree, axis=0)

        decay = np.logspace(-1, 2, num=16)
        bottom = np.zeros(len(decay))
        for i, deg in enumerate(ipc.degs):
            axd['(b)'].bar(decay, ipcs_degree[:,i], width=.3*decay, bottom=bottom, label=deg, color=plt.cm.Spectral_r((deg-1)/ipc.degmax), linewidth=0)
            bottom += ipcs_degree[:,i]
        axd['(b)'].set_xscale('log')
        axd['(b)'].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axd['(b)'].xaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9), numticks=8))
        axd['(b)'].set_ylim(0, 200)
        axd['(b)'].set_xlabel('$\gamma/\Omega$')
        axd['(b)'].set_ylabel(r'$C_{\rm tot}$')
        axd['(b)'].text(-.38, .95, '(b)', transform=axd['(b)'].transAxes)
        axd['(b)'].text(.02, .9, 'MS-QRC', transform=axd['(b)'].transAxes, fontsize=7)

        # ipc of qelm with respect to decay rate
        ipcs_degree = []
        for seed in range(5):
            with open(f'data/decay/ipc_qelm_seed_{seed}/ipc_result.pickle', 'rb') as f:
                ipc_result = pickle.load(f)
            ipc, pkldir, paths = ipc_result['ipc'], ipc_result['pkldir'], ipc_result['paths']
            npzname = f'{pkldir}/state_indicators.npz'
            ipc.get_indicators(npzname, paths, th_scale=th_scale)
            assert np.all(ipc.Ctots<ipc.ranks)
            ipcs_degree.append(ipc.ipcs_degree)
        ipcs_degree = np.mean(ipcs_degree, axis=0)

        decay = np.logspace(-1, 2, 16)
        bottom = np.zeros(len(decay))
        for i, deg in enumerate(ipc.degs):
            axd['(d)'].bar(decay, ipcs_degree[:,i], width=.3*decay, bottom=bottom, label=deg, color=plt.cm.Spectral_r((deg-1)/ipc.degmax), linewidth=0)
            bottom += ipcs_degree[:,i]
        axd['(d)'].set_xscale('log')
        axd['(d)'].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axd['(d)'].xaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9), numticks=8))
        axd['(d)'].set_ylim(0, 200)
        axd['(d)'].set_xlabel('$\gamma/\Omega$')
        axd['(d)'].set_ylabel(r'$C_{\rm tot}$')
        axd['(d)'].text(-.38, .95, '(d)', transform=axd['(d)'].transAxes)
        axd['(d)'].text(.02, .9, 'SS-QRC', transform=axd['(d)'].transAxes, fontsize=7)

        fig.savefig('figures/IPC_decay.pdf', bbox_inches='tight')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--qrc_evolve', action='store_true')
    parser.add_argument('-seed', help="seed for atoms' positions, used with --qrc_evolve", nargs='+', type=int)
    parser.add_argument('--qelm_evolve', action='store_true')
    parser.add_argument('--calculate_ipc', action='store_true')
    parser.add_argument('--plot_ipc', action='store_true')
    args = parser.parse_args()

    main(args)