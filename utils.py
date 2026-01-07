# This script provides some utility functions for the project.

import traceback
from collections import namedtuple
from collections.abc import Callable
from typing import Union, Optional
from itertools import combinations, chain, product
from datetime import datetime
from functools import wraps

import numpy as np
import numpy.typing as npt
from qutip import Qobj, basis, sigmax, sigmaz, fock_dm, tensor # type: ignore
from qutip_qip.operations import expand_operator

Reservoir_cf = namedtuple('reservoir_cf', ['warmup_size', 'train_size', 'validate_size', 'test_size', 'tau', 'Delta0', 'a', 'lattice', 'sigma', 'decay'])
reservoir_cf = Reservoir_cf(warmup_size=200, train_size=400, validate_size=200, test_size=200, tau=10, Delta0=-.5, a=1., lattice='triangle', sigma=.04, decay=.95238)

def position(n_qubits: int, a: float, type: str, sigma: float, seed: int) -> npt.NDArray:
    '''Args:
        a: lattice constant, in units of Rydberg blockade radius.
        type: str, the type of the lattice, could be one of "chain", "square", "triangle".
        sigma: standard deviation of disorder in position, in units of Rydberg blockade radius.
    '''
    def chain1d(n_qubits: int, a: float, xshift: float = 0., yshift: float = 0.) -> npt.NDArray:
        '''Return the position np.array([[x1, y1], ..., [xn, yn]]) of 1d chain of atoms with lattice constant a.'''
        return np.hstack((a*np.arange(n_qubits).reshape(n_qubits, 1)+xshift, np.zeros((n_qubits, 1))+yshift))

    match type:
        case 'chain': # 1 row
            pos = chain1d(n_qubits, a)

        case 'square': # 3 rows, square lattice
            n0 = n_qubits // 3
            n2 = n_qubits // 3
            n1 = n_qubits - n0 - n2
            temp0 = chain1d(n0, a)
            temp1 = chain1d(n1, a, yshift=a)
            temp2 = chain1d(n2, a, yshift=2*a)
            pos = np.vstack((temp0, temp1, temp2))

        case 'triangle': # 3 rows, triangle lattice
            n0 = n_qubits // 3
            n2 = n_qubits // 3
            n1 = n_qubits - n0 - n2
            temp0 = chain1d(n0, a)
            temp1 = chain1d(n1, a, xshift=-a/2, yshift=np.sqrt(3)/2*a)
            temp2 = chain1d(n2, a, yshift=np.sqrt(3)*a)
            pos = np.vstack((temp0, temp1, temp2))
        
        case _:
            raise ValueError('i should be one of {"chain", "square", "triangle"}.')
    
    rng = np.random.default_rng(seed)
    pos += rng.normal(0, sigma, size=pos.shape) # disorder in position to break the spatial symmetry.
    return pos

def hamiltonian(Delta: Union[float, np.ndarray], pos: np.ndarray, tlist: np.ndarray = None) -> Union[Qobj, list]:
    '''
    Args:
        pos: 2darray of shape (n_qubits, 2), each row corresponds to the position of an atom in the x-y plane.
        C6: float, coupling strength of two atoms at a distance of unit length.
    '''
    Omega = 1
    n_qubits = len(pos)
    J = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            R = np.linalg.norm(pos[i] - pos[j])
            J[i, j] = Omega / R**6

    if isinstance(Delta, float): # time-independent Hamiltonian encoded with 1d input time series.
        H = 0
        for i in range(n_qubits):
            H += Delta*expand_operator(fock_dm(2, 0), dims=[2]*n_qubits, targets=i) + Omega/2*expand_operator(sigmax(), dims=[2]*n_qubits, targets=i)
            for j in range(i+1, n_qubits):
                H += J[i, j] * expand_operator(tensor([fock_dm(2, 0)]*2), dims=[2]*n_qubits, targets=[i, j])
    
    elif isinstance(Delta, (np.ndarray, list)) and tlist is None: # time-independent Hamiltonian encoded with historical input time series.
        assert len(Delta) == n_qubits
        H = 0
        for i in range(n_qubits):
            H += Delta[i]*expand_operator(fock_dm(2, 0), dims=[2]*n_qubits, targets=i) + Omega/2*expand_operator(sigmax(), dims=[2]*n_qubits, targets=i)
            for j in range(i+1, n_qubits):
                H += J[i, j] * expand_operator(tensor([fock_dm(2, 0)]*2), dims=[2]*n_qubits, targets=[i, j])

    elif isinstance(Delta, (np.ndarray, list)) and tlist is not None: # time-dependent Hamiltonian encoded with 1d input time series.
        assert len(Delta) == len(tlist)
        Delta = Delta.reshape(-1)

        H0 = Omega/2 * sum([expand_operator(sigmax(), dims=[2]*n_qubits, targets=i) for i in range(n_qubits)]) + sum([J[i, j]*expand_operator(tensor([fock_dm(2, 0)]*2), dims=[2]*n_qubits, targets=[i, j]) for i in range(n_qubits) for j in range(i+1, n_qubits)])
        H1 = sum([expand_operator(fock_dm(2, 0), dims=[2]*n_qubits, targets=i) for i in range(n_qubits)])
        H = [H0, [H1, Delta]]
    return H

def collapse_ops(n_qubits: int, decay: float = 1.) -> list[Qobj]:
    def jump_op(n_qubits: int, i: int, decay: float = 1.) -> Qobj:
        '''Jump operator of the ith qubit.'''
        gamma = decay # in unit of global detuning Omega
        alpha, beta = .025, .08
        op = np.sqrt(gamma)*basis(2, 1) * (alpha*basis(2, 0) + beta*basis(2, 1)).dag()
        return expand_operator(op, dims=[2]*n_qubits, targets=i)
    
    c_ops = [jump_op(n_qubits, i, decay) for i in range(n_qubits)]
    return c_ops

def expect_ops(n_qubits: int, k: Optional[int] = None) -> list[Qobj]:
    if k is None:
        idx = chain.from_iterable(combinations(range(n_qubits), ord) for ord in range(1, n_qubits+1))
        e_ops = [expand_operator(sigmaz(), dims=[2]*n_qubits, targets=idx) for idx in idx]
    
    else: # up to k-local observables of sigmax and sigmaz
        sigma = [sigmax(), sigmaz()]
        e_ops = [expand_operator(tensor(ops), dims=[2]*n_qubits, targets=idx) for ord in range(1, k+1) for idx in combinations(range(n_qubits), ord) for ops in product(sigma, repeat=ord)]
    return e_ops


###############################################################################################
############################################ For timing #######################################
###############################################################################################
def timing(main: Callable) -> Callable:
    @wraps(main)
    def inner(*args, **kwargs):
        start_time = datetime.now()
        try:
            main(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
        end_time = datetime.now()
        print(f'--- Started at {start_time}, ended at {end_time}, the total execution time is {end_time-start_time} ---')
    return inner


###############################################################################################
########################################### For plotting ######################################
###############################################################################################
mpl_rcParams = {
    'font.size': 8,               # match PRL body text (~10pt LaTeX)
    'axes.labelsize': 8,
    'axes.labelpad': 2,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,        # thinner axes for compact plots
    'lines.linewidth': 1.0,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'mathtext.fontset': 'cm',     # Computer Modern for LaTeX match
}

figwidth_sc = 3.375 # figure width for single-column papers
figwidth_dc = 7     # figure width for double-column papers

def trim_zeros(x, pos):
    return ('%g' % x)