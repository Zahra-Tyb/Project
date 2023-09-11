
"""
Self Consistent Born Approximation (SCBA) for disordered 2D Nodal-Line Semimatals (NLSMs)

Author: Zahra Tayebi

This script calculates different components of the self-energy in a quantum system using the
Self Consistent Born Approximation. The system is described by a Hamiltonian, and the
script iteratively computes the self-energy components.

Parameters:
    ep0: Epsilon_0 in the Hamiltonian.
    ec: Epsilon_c, the energy cutoff.
    eta: A small value used to prevent division by zero.
    u: Interaction strength.
    delta: Small energy gap (Delta) in the Hamiltonian.
    c: A counter variable.
    si: Initial value of Sigma_I.
    sx: Initial value of Sigma_x.
    sz: Initial value of Sigma_z.
"""

import numpy as np
import matplotlib.pyplot as plt


# Constants
ep0 = 1                         
xc = 40
eta = 0.01
u = 0.01
delta = 0.2
c = 0

# Initial values of self_energies
si_initial = 0.001 + 1j * 0.001
sx_initial = 0.001 + 1j * 0.001
sz_initial = 0.001 + 1j * 0.001

# Frequency range
W = np.arange(-2, 2.01, 0.01)

# Lists to store results
re_si = []
im_si = []
re_sx = []
im_sx = []

# Perform the self-consistent Born approximation
for w in W:
    c += 1

    for i in range(10000):
        term1 = np.sqrt((w + 1j * eta - si_initial)**2 - (delta + sz_initial)**2) - xc + ep0 - sx_initial
        term2 = np.sqrt((w + 1j * eta - si_initial)**2 - (delta + sz_initial)**2) + ep0 - sx_initial
        term3 = np.sqrt((w + 1j * eta - si_initial)**2 - (delta + sz_initial)**2) + xc - ep0 + sx_initial
        term4 = np.sqrt((w + 1j * eta - si_initial)**2 - (delta + sz_initial)**2) - ep0 + sx_initial

        si = u * ((w + 1j * eta - si_initial) / (np.sqrt((w + 1j * eta - si_initial)**2 - (delta + sz_initial)**2))) * (
                np.log(term2) + np.log(term3) - np.log(term1) - np.log(term4))
        sx = u * (np.log(term2) - np.log(term3) - np.log(term1) + np.log(term4))
        sz = delta * (si_initial) / (w + 1j * eta - 2 * si_initial)

        # Check for convergence
        if np.abs(np.linalg.norm(si) - np.linalg.norm(si_initial)) < 0.0000001 and np.abs(
                np.linalg.norm(sx) - np.linalg.norm(sx_initial)) < 0.0000001:
            break
        else:
            si_initial = si
            sx_initial = sx

    # Store results
    re_si.append(np.real(si))
    im_si.append(np.imag(si))
    re_sx.append(np.real(sx))
    im_sx.append(np.imag(sx))





# Plotting
# First (Real Parts of Self-Energy)
plt.figure(figsize=(8, 6))
plt.plot(W, re_si, 'b')
plt.plot(W, re_sx, 'r')
plt.xlabel(r'$\omega$')
plt.ylabel(r'Re[$\Sigma$]')
plt.legend(['$\Sigma_I$', '$\Sigma_x$'], loc='upper right')
plt.title('Real Parts of Self-Energy')

# Second (Imaginary Parts of Self-Energy)
plt.figure(figsize=(8, 6))
plt.plot(W, im_si, 'b')
plt.plot(W, im_sx, 'r')
plt.xlabel(r'$\omega$')
plt.ylabel(r'Im[$\Sigma$]')
plt.legend(['$\Sigma_I$', '$\Sigma_x$'], loc='upper right')
plt.title('Imaginary Parts of Self-Energy')

plt.show()

