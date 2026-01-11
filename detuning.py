# This script evaluates the IPC with respect to the global bias detuning Delta0. The results are shown in Fig. 3 (a) and (c).

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
from matplotlib.ticker import MultipleLocator

from ipc.information_processing_capacity import single_input_ipc
from utils import timing, hamiltonian, position, reservoir_cf, collapse_ops, expect_ops
from utils import mpl_rcParams, figwidth_sc
mpl.rcParams.update(mpl_rcParams)

def qrc_evolve(seed, decay, Delta0, verbose=False):
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

def qelm_evolve(seed, decay, Delta):
    n_qubits = 10
    assert n_qubits==len(Delta)
    H = hamiltonian(Delta, position(n_qubits, reservoir_cf.a, reservoir_cf.lattice, sigma=reservoir_cf.sigma, seed=seed))
    rho0 = tensor([fock_dm(2, 1)]*n_qubits)
    tlist = [0, reservoir_cf.tau]
    c_ops = collapse_ops(n_qubits, decay)
    e_ops = expect_ops(n_qubits, k=2)
    result = mesolve(H, rho0, tlist, c_ops, e_ops, options=dict(store_states=True, nsteps=int(1e8)))
    X = np.array(result.expect).T[-1]
    return X

def calculate_ipc(zeta, X, dir: str):
    '''zeta should be in [-1,1] for legendre polynomials.'''
    Two = 200
    T = 1000 - Two
    degdelays = [[1, Two], [2, 180], [3, 100], [4, 50], [5, 20], [6, 15]]
    ipc = single_input_ipc(zeta, Two, degdelays, poly='legendre', distr='uniform', zerobased=False)

    paths = [] # Paths for loading data
    for i in range(len(X)):
        x = np.array(X[i]).T[:, :len(zeta)]
        ipc.svd(x)
        path = f'{dir}/pkldir/index_{i}'
        ipc.save_config(path)
        paths.append(path)
        Ctot = 0
        for deg, delay in degdelays:
            ipcs, surs = ipc.compute(deg, delay)
            truncated = ipc.threshold(ipcs, surs, deg, delay, th_scale=1.35)
            Ctot_deg = np.sum(truncated['ipcs'].values)
            print('deg', deg, 'delay', delay, 'Ctot(d)', Ctot_deg)
            Ctot += Ctot_deg
        print('index', i, 'degs', ipc.degs, 'Ctot', Ctot, 'rank', ipc.rank, '\n')

    ipc_result = dict(ipc=ipc, pkldir=f'{dir}/pkldir', paths=paths)
    with open(f'{dir}/ipc_result.pickle', 'wb') as f:
        pickle.dump(ipc_result, f)

@timing
def main(args):    
    path = Path('data/detuning')
    path.mkdir(parents=True, exist_ok=True)
    u = np.load('data/u.npy')
    Delta0 = np.linspace(-9, 3, 13)

    if args.qrc_evolve:
        with MPIPoolExecutor() as executor:
            for seed in tqdm(range(5)):
                verbose = [False]*(len(Delta0)-1) + [True]
                X = list(executor.map(partial(qrc_evolve, seed, reservoir_cf.decay), Delta0, verbose))
                np.save(path/f'X_qrc_seed_{seed}.npy', X)
    
    if args.qelm_evolve:
        with MPIPoolExecutor() as executor:
            n_qubits = 10
            u_pad = np.hstack(([0]*(n_qubits-1), u))
            
            for seed in tqdm(range(5)):
                X = []
                for D in tqdm(Delta0):
                    Delta = D + .1*np.array([u_pad[i:i+n_qubits] for i in range(len(u))])
                    X.append(list(executor.map(partial(qelm_evolve, seed, reservoir_cf.decay), Delta)))
                np.save(path/f'X_qelm_seed_{seed}.npy', X)
    
    if args.calculate_ipc:
        for seed in range(5):
            X = np.load(path/f'X_qrc_seed_{seed}.npy')
            calculate_ipc(u, X, dir=(path/f'ipc_qrc_seed_{seed}').as_posix())

            X = np.load(path/f'X_qelm_seed_{seed}.npy')
            calculate_ipc(u, X, dir=(path/f'ipc_qelm_seed_{seed}').as_posix())

    if args.plot_ipc:
        fig, axd = plt.subplot_mosaic([['(a)'], ['(c)']], layout='constrained', figsize=(figwidth_sc*.5, figwidth_sc*.75))
        r = np.mean(np.load('data/phase/r.npy'), axis=0)
        th_scale = 1.395

        # ipc of qrc with respect to detuning
        ipcs_degree = []
        for seed in range(5):
            with open(f'data/detuning/ipc_qrc_seed_{seed}/ipc_result.pickle', 'rb') as f:
                ipc_result = pickle.load(f)
            ipc, pkldir, paths = ipc_result['ipc'], ipc_result['pkldir'], ipc_result['paths']
            npzname = f'{pkldir}/state_indicators.npz'
            ipc.get_indicators(npzname, paths, th_scale=th_scale)
            assert np.all(ipc.Ctots<ipc.ranks)
            ipcs_degree.append(ipc.ipcs_degree)
        ipcs_degree = np.mean(ipcs_degree, axis=0)

        detuning = np.linspace(-9, 3, 13)
        bottom = np.zeros(len(detuning))
        for i, deg in enumerate(ipc.degs):
            axd['(a)'].bar(detuning, ipcs_degree[:, i], width=.6, bottom=bottom, label=deg, color=plt.cm.Spectral_r((deg-1)/ipc.degmax), linewidth=0)
            bottom += ipcs_degree[:, i]
        axd['(a)'].xaxis.set_major_locator(MultipleLocator(3))
        axd['(a)'].xaxis.set_minor_locator(MultipleLocator(1))
        ax_twinx = axd['(a)'].twinx()
        ax_twinx.plot(np.linspace(-9, 3, 201), r[67], 'k--', alpha=.7)
        ax_twinx.yaxis.set_major_locator(MultipleLocator(.03))
        ax_twinx.set_ylabel(r'$\langle r\rangle$')
        axd['(a)'].set_ylim(0, 200)
        fig.legend(loc='outside lower center', bbox_to_anchor=(0.55, -.08), frameon=False, ncol=6, handlelength=1.0, handleheight=0.2, handletextpad=0.5, columnspacing=1.0)
        axd['(a)'].set_xlabel('$\Delta/\Omega$')
        axd['(a)'].set_ylabel(r'$C_{\rm tot}$')
        axd['(a)'].text(-.38, .95, '(a)', transform=axd['(a)'].transAxes)
        axd['(a)'].text(.02, .9, 'MS-QRC', transform=axd['(a)'].transAxes, fontsize=7)

        # ipc of qelm with respect to detuning
        ipcs_degree = []
        for seed in range(5):
            with open(f'data/detuning/ipc_qelm_seed_{seed}/ipc_result.pickle', 'rb') as f:
                ipc_result = pickle.load(f)
            ipc, pkldir, paths = ipc_result['ipc'], ipc_result['pkldir'], ipc_result['paths']
            npzname = f'{pkldir}/state_indicators.npz'
            ipc.get_indicators(npzname, paths, th_scale=th_scale)
            assert np.all(ipc.Ctots<ipc.ranks)
            ipcs_degree.append(ipc.ipcs_degree)
        ipcs_degree = np.mean(ipcs_degree, axis=0)

        detuning = np.linspace(-9, 3, 13)
        bottom = np.zeros(len(detuning))
        for i, deg in enumerate(ipc.degs):
            axd['(c)'].bar(detuning, ipcs_degree[:, i], width=.6, bottom=bottom, label=deg, color=plt.cm.Spectral_r((deg-1)/ipc.degmax), linewidth=0)
            bottom += ipcs_degree[:, i]
        axd['(c)'].xaxis.set_major_locator(MultipleLocator(3))
        axd['(c)'].xaxis.set_minor_locator(MultipleLocator(1))
        ax_twinx = axd['(c)'].twinx()
        ax_twinx.plot(np.linspace(-9, 3, 201), r[67], 'k--', alpha=.7)
        ax_twinx.yaxis.set_major_locator(MultipleLocator(.03))
        ax_twinx.set_ylabel(r'$\langle r\rangle$')
        axd['(c)'].set_ylim(0, 200)
        axd['(c)'].set_xlabel('$\Delta/\Omega$')
        axd['(c)'].set_ylabel(r'$C_{\rm tot}$')
        axd['(c)'].text(-.38, .95, '(c)', transform=axd['(c)'].transAxes)
        axd['(c)'].text(.02, .9, 'SS-QRC', transform=axd['(c)'].transAxes, fontsize=7)

        fig.savefig('figures/IPC_detuning.pdf', bbox_inches='tight')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--qrc_evolve', action='store_true')
    parser.add_argument('--qelm_evolve', action='store_true')
    parser.add_argument('--calculate_ipc', action='store_true')
    parser.add_argument('--plot_ipc', action='store_true')
    args = parser.parse_args()
    
    main(args)