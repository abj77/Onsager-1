import numpy as np
from onsager import OnsagerCalc
from onsager import crystal
from onsager import crystalStars as stars
from scipy.constants import physical_constants
kB = physical_constants['Boltzmann constant in eV/K'][0]

#a= 3.2342373809
#c_a= 1.5989108537

a=1.0 # 3.2342373809
c_a=np.sqrt(8/3) # 1.5989108537 #
c=a*c_a
HCP = crystal.Crystal.HCP(a0=a,c_a=c_a, chemistry="Zr")
meta_basis = HCP.Wyckoffpos(np.array([5/6,2/3,0.25]))
basis = HCP.basis[0] + meta_basis
crys = crystal.Crystal(HCP.lattice, basis[0:8], chemistry=["Zr"], noreduce=True)
sitelist = crys.sitelist(0)
vacancyjumps = crys.jumpnetwork(0, 1.01*a)
meta_sites = np.arange(2,8,1)
for pos,jlist in enumerate(vacancyjumps):
        if np.any([np.allclose(dx,[0.5, -0.8660254, 0.]) for (i,j), dx in jlist]):
            ind1 = pos
            break
#print("ind1 = ",ind1)
for pos,jlist in enumerate(vacancyjumps):
        if np.any([np.allclose(dx,[ 0.25, -0.4330127, 0.]) for (i,j), dx in jlist]):
            ind2 = pos
            break
#print("ind2 = ",ind2)
jumpnetwork = [vacancyjumps[1], vacancyjumps[ind2]]
jumpnetwork2 = [vacancyjumps[1], vacancyjumps[ind1]]
starset = stars.StarSetMeta(jumpnetwork, crys, 0, meta_sites = meta_sites)
starset.generate(2)
to_del = []
##for state in starset.states:
##    if state.i not in meta_sites and state.j in meta_sites:
##        if np.allclose(0.866025403784,np.linalg.norm(state.dx)):
##            to_del.append(state)
HCPdiffuser = OnsagerCalc.VacancyMediatedMeta(crys, 0, sitelist, jumpnetwork, 4, meta_sites = np.arange(2,8,1), jumpnetwork2= jumpnetwork2, deleted_states=to_del)

HCPtracer = {'preV': np.array([1.0,1.0/10000000.0]), 'eneV': np.array([0.0, 0.0]),
             'preT0': np.array([ 0.5, 1.0]),
             'eneT0': np.array([0, 0]),
              }

##HCPtracer = {'preV': np.array([1.0,54.169024409/18.299152044],), 'eneV': np.array([0.0,0.51727]),
##             'preT0': np.array([ 54.169024409/9.26073917, 54.169024409/10.40702378]),
##             'eneT0': np.array([0.613339999999994,0.553549999999973]),
##              }

HCPtracer.update(HCPdiffuser.maketracerpreene(**HCPtracer))
HCPtracer['preT2'] = np.array([ 0.5, 0.5])
print(HCPtracer)
#HCPtracer['preT2'] = np.array([ 54.169024409/9.26073917, 0.5* 54.169024409/10.40702378])
#HCPtracer['preT2'] = np.array([ 54.169024409/9.26073917, 0.5* 54.169024409/10.40702378,54.169024409/10.40702378])
#HCPtracer['eneT2'] = np.array([0.613339999999994,0.553549999999973,1e12])

#Temp=np.arange(500, 1010, 100)
#Temp = [500,1000]
Temp = [1000]
D_Onsager=[]
for T in Temp:
    pre=1e-8 # THz and Angstrom unit scaling
    Lvv, Lss, Lsv, L1vv = HCPdiffuser.Lij(*HCPdiffuser.preene2betafree(kB*T, **HCPtracer))
    Lss=Lss*pre
    #D_Onsager.append([Lss[0,0],Lss[1,1],Lss[2,2]])
    print(T, Lss[0,0],Lss[1,1],Lss[2,2])
    print(T, Lvv[0,0],Lvv[1,1],Lvv[2,2])
    print(T, Lsv[0,0]/Lvv[0,0],Lsv[1,1]/Lvv[1,1],Lsv[2,2]/Lvv[2,2])

D_Onsager=np.array(D_Onsager)