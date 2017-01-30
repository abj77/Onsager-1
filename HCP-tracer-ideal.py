

import numpy as np
from onsager import OnsagerCalc
from onsager import crystal
from scipy.constants import physical_constants
kB = physical_constants['Boltzmann constant in eV/K'][0]
a= 1.0 # 3.2342373809
c_a=  np.sqrt(8/3)
#c_a= 1.5989108537 # np.sqrt(8/3)
c=a*c_a
HCP = crystal.Crystal.HCP(a0=a,c_a=c_a, chemistry="Zr")
sitelist = HCP.sitelist(0)
vacancyjumps = HCP.jumpnetwork(0, 1.01*a)
HCPdiffuser = OnsagerCalc.VacancyMediated(HCP, 0, sitelist, vacancyjumps, 1)
##
##HCPtracer = {'preV': np.array([1.0]), 'eneV': np.array([0.0]),
##             'preT0': np.array([ 0.5*54.169024409/10.40702378, 54.169024409/9.26073917]),
##             'eneT0': np.array([0.553549999999973,0.613339999999994]),
##              }
HCPtracer = {'preV': np.array([1.0]), 'eneV': np.array([0.0]),
             'preT0': np.array([0.5, 0.5]),
             'eneT0': np.array([0,0]),
              }
HCPtracer.update(HCPdiffuser.maketracerpreene(**HCPtracer))
HCPtracer['preT2'] = np.array([0.5,0.5])
#Temp=np.arange(500,1023, 100)
Temp = [1000]
D_Onsager=[]
for T in Temp:
    pre=1e-8 # THz and Angstrom unit scaling
    Lvv, Lss, Lsv, L1vv = HCPdiffuser.Lij(*HCPdiffuser.preene2betafree(kB*T, **HCPtracer))
    Lss=Lss*pre
    D_Onsager.append([Lss[0,0],Lss[1,1],Lss[2,2]])
    print(T, Lss[0,0],Lss[1,1],Lss[2,2])
    print(T, Lvv[0,0],Lvv[1,1],Lvv[2,2])
    print(T, Lsv[0,0]/Lvv[0,0],Lsv[1,1]/Lvv[1,1],Lsv[2,2]/Lvv[2,2])
