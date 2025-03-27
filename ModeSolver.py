#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:18:18 2025

@author: rhyscounsell
"""


import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy.integrate import quad

#Units
eV = 1.602176634e-12 # electron volt [erg]

eps_MeV_to_CGS = 1.7827e12
pres_MeV_to_CGS = 1.6022e33

length_geometric_to_CGS = 1e5
pressure_geometric_to_CGS = 1.2102555643382062e+39
density_geometric_to_CGS = 1.3465909215001088e+18
mass_geometric_to_CGS = 1.3465909215001088e+33
BaryonMass = 1.67262192369e-24/mass_geometric_to_CGS

mass_geometric_to_Msol = 0.6772199503624046
time_geometric_to_CGS = 3.3356409519815205e-06

pressure_geometric_to_natural=755382.1087233546


class Star:
    """Structure of relativistic star.

    Parameters
    ----------
    eos: object
        EOS model with functions:
        PofE(epsilon) pressure as a function of energy density
        EofP(p) energy density as a function of pressure
        Gamma1(p) adiabatic index gamma_1 as a function of pressure
        DpDepsofP(p) derivative of pressure w.r.t. energy density as a function of pressure
        
    epsc : float
        Energy Denisty [km^-2] at stellar centre
        
    l: int
        
    Notes
    -----
    Assumes geometric units, where G = c = 1.
    """
    def __init__(self, eos,epsilonc, l):
        
        self.l = l
        self.atol = 1e-10
        self.rtol = 1e-10
        self.epsilonc = epsilonc
        self.r0 = 1e-5
        self.eos=eos
        pc=self.eos.PofE(self.epsilonc)
        
        # Taylor expansion near centre for accuracy
        p2 = - 4*np.pi/3*(self.epsilonc + pc)*(self.epsilonc + 3*pc)
        m3 = 4*np.pi*self.epsilonc

        m0 = 1/3*self.r0**3*m3
        p0 = pc + 1/2*self.r0**2*p2
        
        self.__sol = solve_ivp(self.structure, [self.r0, 20], [m0, p0, 0],
                               method='DOP853', dense_output=True, t_eval=np.linspace(self.r0,20,100000),
                               events=self.surface, rtol=self.rtol, atol=self.atol)
        #print(self.__sol.success, self.__sol.message)

        self.rsol = self.__sol.t
        self.msol, self.psol, nusol = self.__sol.y
        self.M, self.R = self.msol[-1], self.rsol[-1]
        print('R = {} Km'.format(self.R))
        print('M = {} M\u2609'.format(self.M*0.6772199503624046))
        #print(simpson(self.BaryonM(self.rsol),self.rsol)*mass_geometric_to_Msol)
        
        # adjust to match surface boundary condition
        self.nusol = nusol - nusol[-1] + np.log(1 - 2*self.M/self.R)

        # calculate central metric potential
        nu2 = 8*np.pi/3*(self.epsilonc + pc)
        self.nuc = self.nusol[0] - self.r0**2*nu2/2

        # interpolate metric potential separately
        self.nu = CubicSpline(self.rsol, self.nusol, extrapolate=False)
        
        epsmade=[]
        presmade=[]
        rss=self.rsol
        
        for i in range(len(rss)):
            epsmade.append(self.eos.EofP(self.p(rss[i])))
            presmade.append(self.p(rss[i]))
        
        
        depsdr = np.gradient(epsmade,rss,edge_order=2)
        
        self.depsdr = interp1d(rss,depsdr,fill_value='extrapolate')
        
    def structure(self, r, y):
        """Stellar structure equations.

        Parameters
        ----------
        r : float
            Radial coordinate [km].
        y : (3,) array_like
            Mass `m` [km], pressure `p` [km^-2] and metric potential `nu`
            [dimensionless] at `r`.

        Returns
        -------
        dmdr : float
            Derivative of mass `m` [dimensionless] at `r`.
        dpdr : float
            Derivative of pressure `p` [km^-3] at `r`.
        dnudr : float
            Derivative of metric potential `nu` [km^-1] at `r`.
        """
        m, p, nu = y

        if p<0:
            epsilon = np.nan
        else:
            epsilon=self.eos.EofP(p)

        dmdr = 4*np.pi*r**2*epsilon
        dnudr = 2*(m + 4*np.pi*r**3*p)/(r*(r - 2*m))
        dpdr = - (epsilon + p)*dnudr/2

        return [dmdr, dpdr, dnudr]

    def surface(self, r, y):
        """Definition of stellar surface: `p = 0`."""
        m, epsilon, nu = y
        p = self.eos.PofE(epsilon)
        return p
    
    surface.terminal = True
    surface.direction = -1
    
    def m(self, r):
        """Return mass.

        Parameters
        ----------
        r : float
            Radial coordinate [km].

        Returns
        -------
        m : float
            Mass `m` [km] at `r`.
        """
        m, p, nu = self.__sol.sol(r)
        return m

    def p(self, r):
        """Return pressure.

        Parameters
        ----------
        r : float
            Radial coordinate [km].

        Returns
        -------
        p : float
            Pressure `p` [km^-2] at `r`.
        """
        m, p, nu = self.__sol.sol(r)
        return p
    
    def epsofp(self,p):
        epsofp = self.eos.EofP(p)
        return epsofp
    
    def dpdeps(self,p):
        dpdeps = self.eos.DpDepsofP(p)
        return dpdeps
    
    def dpdr(self,r):
        m, p, nu = self.__sol.sol(r)
        eps = self.epsofp(p)
        dpdr = -(eps+p)*(m+4*np.pi*r**3*p)/(r*(r-2*m))
        return dpdr
    
    def lam(self,r):
        m = self.m(r)
        lam = - np.log(1-2*m/r)
        return lam
           
    def U(self,r):
         m, p, nu = self.__sol.sol(r)
         Epsilon = self.epsofp(p)
         U = 4*np.pi*r**3*Epsilon/m
         return U
     
    def V(self,r):
         m, p, nu = self.__sol.sol(r)
         Epsilon = self.epsofp(p)
         V = (Epsilon+p)*(m+4*np.pi*r**3*p)/(p*(r-2*m))
         return V
    
    def beta(self,r):
        m, p, nu = self.__sol.sol(r)
        nu = self.nu(r)
        lam = self.lam(r)
        beta = np.exp(nu+lam/2)*(1+4*np.pi*r**3*p/m)
        return beta
    
    def Aplus(self,r):
        m, p, nu = self.__sol.sol(r)
        Epsilon = self.epsofp(p)
        depsdr  = self.depsdr(r)
        dpdr = self.dpdr(r)
        gamma = self.eos.Gamma(p)
        Aplus = np.exp(-self.lam(r)/2)*(depsdr/(p+Epsilon)-dpdr/(gamma*p))
        return Aplus
    
    def Aminus(self,r):
        m, p, nu = self.__sol.sol(r)
        Epsilon = self.epsofp(p)
        depsdr  = self.depsdr(r)
        dpdr = self.dpdr(r)
        gamma = self.eos.Gamma(p)
        Aminus = np.exp(-self.lam(r)/2)*(depsdr/(p+Epsilon)+dpdr*(2/(p+Epsilon)-1/(gamma*p)))
        return Aminus
     
    def f(self, r, Z, omega):
        """
        The system of equations that describe a mode with degree `l` and 
        (dimensionless) squared frequency `omega_tilde^2`. Formally, this is a 
        linear, fourth-order system of coupled ordinary differential equations, 
        given by the matrix equation `dZ/dx = Q Z`.

        Parameters
        ----------

        x : float
            Dimensionless radius
        Z : array
            Abstract vector field (z_1, z_2)
        omega2_tilde : float
            Dimensionless squared frequency
        
        Returns
        -------

        dZ_dx : array
            Derivative of abstract vector field 
            (dz_1_dx, dz_2_dx)

        """
        
        
        m = self.m(r)
        p = self.p(r)
        lam=self.lam(r)
        u=self.U(r)
        v=self.V(r)
        beta=self.beta(r)
        Am=self.Aminus(r)
        Ap=self.Aplus(r)
        l = self.l
        Gamma = self.eos.Gamma(p)
        omega2 = omega**2
        
        Q = np.array([
                [v/Gamma - l - 1,l*(l+1)*m*np.exp(lam/2)/(omega2*r**3)-v/(Gamma*beta)], 
                [np.exp(lam/2)*(omega2*r**3/m+Ap*r*beta),3-u-Am*np.exp(lam/2)*r-l]
                
        ]) / r

        dZ_dr = np.dot(Q, Z)
        
        return dZ_dr      

    def boundary_conditions_centre(self,r,z_1, omega):
        
        m, p, nu = self.__sol.sol(r)
        omega2 = omega**2
        z_2=z_1/(self.l*m/(omega2*r**3))

        return z_2

    def boundary_conditions_surface(self,r,z_1):
        
        z_2=z_1*self.beta(r)
        
        return z_2     

    
    def g(self,omega_guess):
            
        R = self.R
        
        
        # Linearly independent solution 1
        z_1 = 1
        
        # Boundary conditions at small r
        z_2 = self.boundary_conditions_centre(self.r0,z_1, omega_guess)
        
        # Solutions for 0 < r <= R/2
        xs = np.linspace(self.r0, self.R/2, 100000)
        
        # Integrate from small r to r = R/2
        sol_1 = solve_ivp(self.f, [xs[0], xs[-1]], [z_1, z_2], 
                          method= 'BDF', 
                          args=(omega_guess,), t_eval=xs,
                          rtol=1e-10, atol=1e-10
                          )
        #print(sol_1.success, sol_1.message)
        
        # Linearly independent solution 2
        z_1=1

        # Boundary conditions at r = R
        z_2 = self.boundary_conditions_surface(R,z_1)
        
        # Solutions for R/2 <= r <= R
        xs = np.linspace(R, self.R/2, 100000)

        # Integrate from r = R to r = R/2
        sol_2 = solve_ivp(self.f, [xs[0], xs[-1]], [z_1, z_2], 
                          method='BDF',
                          args=(omega_guess,), t_eval=xs,
                          rtol=1e-10, atol=1e-10
                          )
        #print(sol_2.success, sol_2.message)
        
        P = np.column_stack((sol_1.y[:, -1], -sol_2.y[:, -1]))

        a = np.linalg.det(P)

        return(a)
    
    def k2ODE(self,r,y):
        
        lam = self.lam(r)
        m = self.m(r)
        p = self.p(r)
        eps = self.epsofp(p)
        dpdeps = self.dpdeps(p)
        dnu_dr = 2*(m + 4*np.pi*r**3*p)/(r*(r - 2*m))
        l = star.l
        
        dy_dr = (- (y - 1)*y - (2 + np.exp(lam)*(2*m/r + 4*np.pi*r**2*(p - eps)))*y+ l*(l + 1)*np.exp(lam)
             - 4*np.pi*r**2*np.exp(lam)*(5*eps + 9*p + (eps + p)/(dpdeps))
             + (r*dnu_dr)**2)/r
        
        return dy_dr
    
class mode:
    """
    Solves mode equations
    
    Parameters
    ----------
    star : Object
        Background star
    
    omega_guess : float
        Guess mode freuency, dimensionless

    Notes
    -----
    Assumes geometric units, where G = c = 1.
    """
    
    
    def __init__(self, star, omega_guess):
        self.star = star
        self.omega_guess = omega_guess
        
        R = self.star.R
        
        sol = root_scalar(
                        self.star.g, 
                        bracket=np.array([1 - 0.05, 1 + 0.05])*omega_guess, 
                        method='brentq', xtol=1e-10, rtol=1e-10
        )
        print(sol.converged, sol.flag)
        print('\u03c9\u0303 = {}'.format(sol.root/np.sqrt(self.star.M/self.star.R**3)))
        print('\u03c9 = {} Hz'.format(sol.root/(2*np.pi)/time_geometric_to_CGS))
         
        omega= sol.root
        self.root = sol.root
    
        # Solutions for 0 < r <= R/2
        xs = np.linspace(self.star.r0, self.star.R/2, 100000)
        
        # Linearly independent solution 1
        z_1 = 1
        
        # Boundary conditions at small r
        z_2 = self.star.boundary_conditions_centre(self.star.r0,z_1,omega)
        
        # Integrate from small r to r = R
        sol_1 = solve_ivp(self.star.f, [xs[0],xs[-1]], [z_1, z_2], 
                          method= 'BDF', t_eval = xs,args=(omega,),
                          dense_output=True,
                          rtol=1e-10, atol=1e-10
                          )
        #print(sol_1.success, sol_1.message)
        
        # Linearly independent solution 2
        z_1=1

        # Boundary conditions at r = R
        z_2 = self.star.boundary_conditions_surface(R,z_1)
        
        # Solutions for R/2 <= r <= R
        xs = np.linspace(R, self.star.R/2, 100000)

        # Integrate from r = R to r = R/2
        sol_2 = solve_ivp(self.star.f, [xs[0], xs[-1]], [z_1, z_2], 
                          method='BDF',t_eval = xs, args=(omega,), 
                          dense_output=True,
                          rtol=1e-10, atol=1e-10
                          )
        #print(sol_2.success, sol_2.message)
        
        P_tilde = np.column_stack((sol_1.y[:, -1], -sol_2.y[:, -1]))
        P_tilde[0, :] = [0,self.star.R**3]
        
        Z = [1, 0]
        
        X = np.linalg.solve(P_tilde, Z)
        
        #print(X)
        
        # General solution for 0 < r <= R/2
        Z_c = X[0]*sol_1.sol(sol_1.t)
        # General solution for R/2 <= r <= R
        Z_s = X[1]*sol_2.sol(sol_2.t)
        
        # Unique physical solution
        rs = np.concatenate((sol_1.t[:-1], sol_2.t[::-1]))
        
        Z_1s = np.concatenate((Z_c[0, :-1], Z_s[0, ::-1]))
        Z_2s = np.concatenate((Z_c[1, :-1], Z_s[1, ::-1]))
        
        self.sol_1 = sol_1
        self.sol_2 = sol_2
        
        self.Z_1s=Z_1s
        self.Z_2s=Z_2s
        
        Ws=[]
        Vs=[]
        Xir=[]
        for i in range(len(rs)):
            Ws.append(Z_1s[i]*rs[i]**3)
            Vs.append(-self.star.m(rs[i])*Z_2s[i]/(omega**2*rs[i]))
            Xir.append(Z_1s[i]*rs[i]*np.exp(-self.star.lam(rs[i])/2))
    
        
        self.Ws = Ws
        self.Vs = Vs
        self.rs = rs
        self.Xir=Xir
        
     
        plt.figure()
        plt.plot(rs/self.star.R,Ws)
        plt.xlabel('r/R')
        plt.ylabel('W')
        plt.title('$\~\omega = {}$' .format(omega/np.sqrt(star.M/star.R**3)), fontsize=12)
        
        plt.figure()
        plt.plot(rs/self.star.R,Vs)
        plt.xlabel('r/R')
        plt.ylabel('V')
        plt.title('$\~\omega = {}$' .format(omega/np.sqrt(star.M/star.R**3)), fontsize=12)
        
        fig, axs = plt.subplots(2, 2)
        
        axs[0,0].plot(rs/self.star.R, Z_1s)
        axs[0,0].set_ylabel('$Z_1$')
        axs[0,0].set_xlabel('r/R')

        axs[0,1].plot(rs/self.star.R, Z_2s)
        axs[0,1].set_ylabel('$Z_2$')
        axs[0,1].set_xlabel('r/R')
        
        axs[1,0].plot(rs/self.star.R, np.gradient(Z_1s,rs))

        axs[1,0].set_ylabel('$dZ_1/dr$')
        axs[1,0].set_xlabel('r/R')
        
        axs[1,1].plot(rs/self.star.R, np.gradient(Z_2s,rs))
        axs[1,1].set_ylabel('$dZ_2/dr$')
        axs[1,1].set_xlabel('r/R')

        plt.show()
        
        
        dwdr = np.gradient(Ws,rs,edge_order=2)
        dvdr = np.gradient(Ws,rs,edge_order=2)
        
        self.dwdr = interp1d(self.rs,dwdr,fill_value='extrapolate')
        self.dvdr = interp1d(self.rs,dvdr,fill_value='extrapolate')
    
        self.Z1int =  interp1d(self.rs,Z_1s,fill_value='extrapolate')
        self.Z2int =  interp1d(self.rs,Z_2s,fill_value='extrapolate')
        self.Wint =  interp1d(self.rs,Ws,fill_value='extrapolate')
        self.Vint =  interp1d(self.rs,Vs,fill_value='extrapolate')
         
        Q=quad(self.Q,self.star.r0,self.star.R,limit=100)/(star.M*star.R**2)
    
        A2=quad(self.A2,self.star.r0,self.star.R,limit=100)
        
        TQ=(Q[0])/np.sqrt((A2[0])/(star.M*star.R**2))
        print('\u0051\u0303= {:f}' .format(np.abs(TQ)))
        

    def Q(self,r):
        Test = 2*(self.star.eos.EofP(self.star.p(r))+self.star.p(r))*np.exp(self.star.nu(r)/2)*r*(self.Wint(r)-3*r*self.Vint(r)*np.exp(self.star.lam(r)/2))
        return Test
        
    def A2(self,r):
        TestA=np.exp((self.star.lam(r)-self.star.nu(r))/2)*(self.star.epsofp(self.star.p(r))+self.star.p(r))/r**2*(np.exp(self.star.lam(r))*(self.Wint(r))**2+
                                                6*self.Vint(r)**2)
        return TestA
    

#Generate EOS Model
eos = EOS(file)

#Set central energy density    
epsc=9.86e-4 #[km^-2]
l=2

#Create background star
star = Star(eos,epsc,l)

#Compactness
C = star.M/ star.R 

print('C = {}'.format(C))

#Calculate love number k_2
Sol=solve_ivp(star.k2ODE,[star.r0,star.R],[2],method='BDF', 
 dense_output=True,rtol=1e-11, atol=1e-11)
Y=Sol.y[0][-1]
k2 = (8*C**5/5*(1 - 2*C)**2*(2 + 2*C*(Y - 1) - Y)
   *(2*C*(6 - 3*Y + 3*C*(5*Y - 8))
     + 4*C**3*(13 - 11*Y + C*(3*Y - 2) + 2*C**2*(1 + Y))
     + 3*(1 - 2*C)**2*(2 - Y + 2*C*(Y - 1))*np.log(1 - 2*C))**(-1))

print('k\u2082 = {}'.format(k2))

#Dimensionless Tidal Deformability
DTD = 2/3*k2/(C**5)

print('\u039B = {}'.format(DTD))


#Plot the spectrum. A mode corresponds to a zero in the plot

def Spectrum(background,start,end,number):
    """Plots oscillation spectrum.

    Parameters
    ----------
    background: object
        background star
        
    start : float
        dimensionless omega guess
        
    end: float
        dimensionless omega guess
        
    number: int
        number of points to try in interval
    """
    
    a=np.linspace(start,end,number) #Range of dimensionless frequency
    b=[]
    for i in range(len(a)):
        b.append(background.g(a[i]*np.sqrt(star.M/star.R**3)))
        if i%2==0:
            print(i)
    plt.figure()
    plt.plot(a,b,'+')
    plt.xlabel(r'$\tilde \omega$')
    plt.ylabel(r'g($\tilde \omega$)')
    plt.plot(a,np.zeros(len(a)),'--')

Spectrum(star,0.3,0.6,10)

#Solve for a mode
m=mode(star,omega_guess*np.sqrt(star.M/star.R**3))


