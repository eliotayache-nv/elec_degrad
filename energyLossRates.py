# -*- coding: utf-8 -*-
# @Author: elay7268
# @Date:   2021-12-09 22:15:17
# @Last Modified by:   elay7268
# @Last Modified time: 2021-12-16 10:50:27


'''
This code outputs the energy loss rates following Barnes&Kasen2016
(ApJ 829:110, 2016)
Section 3: Thermalisation physics.
Processes by which energetic decay products thermalize in the KN ejecta.
Focus on beta-particles.
'''

import numpy as np
import matplotlib.pyplot as plt

# Constants
lambda_ee_ = 10  # Coulomb logarithm for electron-electron scattering
m_e_ = 9.10938356e-28  # electron mass (g)
r_e_ = 2.8179403227e-13  # classical electron radius (cm)
m_u_ = 1.660539e-24  # nuclear mass unit (g)
c_ = 2.99792458e10  # light speed (cm/s)
kB_ = 8.617333262145e-11  # Boltzmann constant (MeV/K)

delta_ = 1  # power-law profile for inner ejecta (Barnes&Kasen2016)
n_ = 1  # power-law profile for outer ejecta (Barnes&Kasen2016)


def plasma_losses(E_beta, n_e, T):
    ''' Returns beta_particle losses from interaction with free thermal
    electrons (MeV/s).
    args:
    E_beta = beta-particle kinetic energy (MeV)
    n_e = free electron number density (cm-3)
    T = ejecta temperature (MeV)
    '''

    return(7.7e-15 
           * E_beta**(-.5) 
           * n_e * lambda_ee_ 
           * (1. - (3.9 / 7.7) * T / E_beta))


def calc_n_eb(rho, z_over_a):
    ''' Returns the bound electron number density (cm-3) for a given
    average Z/A.
    args:
    rho = mass density (g/cm-3)
    z_over_a = composition (0.4 inner / 0.55 outer ejecta)
    '''
    return(rho / m_u_ * z_over_a)


def IE_losses(E_beta, Ibar, v_beta, n_eb):
    ''' Returns beta_particle losses due to ionization and excitation
    of atomic electrons (MeV/s).
    args:
    E_beta = beta-particle kinetic energy (MeV)
    I_bar = average ionization and excitation potential (MeV) (eq. 9)
    v_beta = beta-particle speed (cm/s)
    n_eb = bound electron number density (cm-3) 
    '''

    tau = E_beta * 1.60218e-6 / (m_e_ * c_**2)  # E_beta converted to erg
    g = 1 + tau**2 / 8 - (2*tau+1) * np.log(2)
    Edot_IE = 2*np.pi * r_e_**2 * m_e_ * c_**3 * n_eb / (v_beta/c_)
    Edot_IE *= (2 * np.log(E_beta/Ibar) 
                + np.log(1 + tau/2) 
                + (1 - (v_beta/c_)**2) * g)

    return(Edot_IE / 1.60218e-6)


def synch_losses(v_beta, B):
    ''' Returns synchrotron losses from Barnes&Kasen2016 eq. 11. (erg/s)
    args:
    v_beta = beta particle velocity
    B = Magnetic field strength
    '''
    gamma = np.sqrt(1./(1-(v_beta/c_)**2))
    loss = (4./9.) * r_e_**2 * c_ * gamma**2 * (v_beta / c_)**2 * B**2
    return(loss)


def calc_mass_density(r, t, E, Mej, delta=delta_, n=n_):
    ''' Returns the mass density for a given r and t in the ejecta.
    args:
    r = radius (cm)
    t = time (s)
    E = Explosion energy (erg)
    M = ejecta mass (Msun)
    '''

    zeta_v = 1
    zeta_rho = 1

    v_t = 7.1e8 * zeta_v * ((E*1.e-51) / (Mej/Msun))**(.5)

    v = r/t
    if v < v_t:
        rho = zeta_rho * (Mej/(v_t*t)**3) \
              * (r / (v_t*t))**(-delta)
    else:
        rho = zeta_rho * (Mej/(v_t*t)**3) \
              * (r / (v_t*t))**(-delta)
    return(rho)


def main():

    E_beta = np.logspace(-2, 1, 100)
    T = 10000 * kB_
    n_e = 1.e21
    Ibar = np.exp(6.4) * 1.e-6  # (MeV)
    gamma_beta = E_beta / (m_e_ * c_**2)
    v_beta = np.sqrt(1 - 1./gamma_beta**2)
    z_over_a = 0.4
    rho = 1.e-12  # computed from basic ejecta composition
    # B = 3.7e-6

    n_eb = calc_n_eb(rho, z_over_a)

    plt.figure()    
    plt.plot(E_beta, plasma_losses(E_beta, n_e, T) / rho)
    plt.plot(E_beta, IE_losses(E_beta, Ibar, v_beta, n_eb) / rho)
    [plt.plot(E_beta, IE_losses(E_beta, Ibar, v_beta, n_eb) / rho) for alpha in np.logspace(-5,5,10)]
    # plt.plot(E_beta, synch_losses(v_beta, B) / rho)
    plt.xscale('log')
    plt.yscale('log')


if __name__ == '__main__':
    main()














