"""
Example of how to solve for oscillation modes in the GR Cowling formalism.
Here we use the BSk22 equation-of-state and functionals obtained from [1][2]
We will calculate the frequencies of the f-mode and first two g-modes 
for a neutron star of mass approx 1.4 solar masses.

[1] Shchechilin N. N., Chamel N., Pearson J. M.; "Unified equations of state for cold nonaccreting neutron stars with 
    Brussels-Montreal functionals. IV. Role of the symmetry energy in pasta phases", 2023, Phys. Rev. C, 108, 025805

[2] Goriely S., Chamel N., Pearson J. M.;"Further explorations of Skyrme-Hartree-Fock-Bogoliubov mass formulas. XIII. 
    The 2012 atomic mass evaluation and the symmetry coefficient", 2013, Phys. Rev. C, 88, 024308
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import ModeSolver as MS

#Import Gamma_1 as a function of baryon density from file
BSk22=pd.read_csv("EOS/GamBSk22.txt", sep='\s+',header=None,names=['a','b'])

#Coefficients of BSk functionals
C1_22=[7.02e8,1.133e11,6.19e7,4.54e6,5.46e5,15.24,0.0683,8.86,4611,48.07,2.697,81.7,7.05,1.5]
C2_22=[6.682,5.651,0.00459,0.14359,2.681,11.972,13.993,1.2904,2.665,-27.787,2.0140,4.09,14.135,28.03,-1.921,1.08,14.89,0.098,11.67,4.75,-0.037,14.1,11.9]

#Units
e_gr = -9.1536 #MeV
e_0 = 930.4118 #MeV

clight = 2.9979e8 #m/s
MeV = 1.602176634e-13
fm = 1e15

Mn = 939.5654 #Mev/c2

pressure_geometric_to_CGS = 1.2102555643382062e+39
density_geometric_to_CGS = 1.3465909215001088e+18
mass_geometric_to_CGS = 1.3465909215001088e+33
BaryonMass = 1.67262192369e-24/mass_geometric_to_CGS

mass_geometric_to_Msol = 0.6772199503624046
time_geometric_to_CGS = 3.3356409519815205e-06


class BSK:
    """BSk EOS
      Based on functionals of
    """

    def __init__(self, ceps, cpres, BSk):
        
        self.ceps = ceps
        self.cpres = cpres
        self.BSk = BSk
        
        nbar=np.logspace(-11,2,100000)
        epsmade = self.epsilon(nbar)
        presmade = self.PofE(epsmade)
        
        self.nbar = nbar
        self.epsmade = epsmade
        self.presmade = presmade
        
        self.nofeps = CubicSpline(epsmade, nbar, extrapolate=False)
        self.EofP = CubicSpline(presmade, epsmade, extrapolate=False)
        
        dpdeps = np.gradient(presmade,epsmade,edge_order=2)
        self.DpDepsofP = CubicSpline(presmade,dpdeps,extrapolate=False)
        
        nbarGamma = self.BSk['a']
        gamma1 = self.BSk['b']
        epsGamma=[]
        for i in range(len(nbarGamma)):
            epsGamma.append(self.epsilon(nbarGamma[i]))
        
        presGamma=[]
        for i in range(len(nbarGamma)):
            presGamma.append(self.PofE(epsGamma[i]))
        
        self.Gamma1 = CubicSpline(presGamma, gamma1, extrapolate=True)
        
    def e_eq(self,n):
        #MeV
        c = self.ceps
        e_eq = e_gr+(c[0]*n)**(7/6)/(1+np.sqrt(c[1]*n))*(1+np.sqrt(c[3]*n))/((1+np.sqrt(c[2]*n))*(1+np.sqrt(c[4]*n)))/(1+c[8]*n)+c[5]*n**c[6]*(
            1+c[7]*n)*(1-1/(1+c[8]*n))/(1+(c[12]*n)**c[13])+(c[9]*n)**c[10]/(1+c[11]*n)*(1-1/1+(c[12]*n)**c[13])
        return e_eq
    
    def epsilon(self,n):
        #[eps] = geometric
        #[n] = fm^-3
        e_eq = self.e_eq(n)
        epsilon = n*(e_eq/(clight**2)+Mn)
        eps2 = epsilon*MeV*fm**3/clight**2*1e3/(1e6)/density_geometric_to_CGS
        return eps2
    
    def Xi(self,eps):
        #[eps] = geometric
        eps2 = eps*density_geometric_to_CGS #g/cm^3
        xi = np.log(eps2)/np.log(10)
        return xi
    
    def PofE(self, eps):
        #[pres] = geometric
        c = self.cpres
        xi = self.Xi(eps)
        logp = (c[0]+c[1]*xi+c[2]*xi**3)/(1+c[3]*xi)/(np.exp(c[4]*(xi-c[5]))+1)+(c[6]+c[7]*xi)/(np.exp(c[8]*(c[5]-xi))+1)+(
            c[9]+c[10]*xi)/(np.exp(c[11]*(c[12]-xi))+1)+(c[13]+c[14]*xi)/(np.exp(c[15]*(c[16]-xi))+1)+c[17]/(1+(c[19]*(xi-c[18]))**2)+c[20]/(
                1+(c[22]*(xi-c[21]))**2)
        p = (10**logp)/pressure_geometric_to_CGS
        return p    
    
#Generate EOS Model
eos = BSK(C1_22,C2_22,BSk22)

#Set central energy density    
epsc=5.1389e-4 #[km^-2]
l=2

#Create background star
star = MS.Star(eos,epsc,l)

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
MS.Spectrum(star,1.2,1.5,10)

#Solve for a mode
#First we solve for the f-mode
m=MS.mode(star,1.43*np.sqrt(star.M/star.R**3))


#Now we will look at the low frequency spectrum
MS.Spectrum(star,0.12,0.2,10)

#Next the first g-mode
m=MS.mode(star,0.182*np.sqrt(star.M/star.R**3))

#Finally the second g-mode
m=MS.mode(star,0.127*np.sqrt(star.M/star.R**3))
