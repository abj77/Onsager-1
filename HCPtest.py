import numpy as np
from onsager import OnsagerCalc
from onsager import crystal
from onsager import crystalStars as stars
from scipy.constants import physical_constants

kB = physical_constants['Boltzmann constant in eV/K'][0]

HCP = crystal.Crystal.HCP(1.)
meta_basis = HCP.Wyckoffpos(np.array([5 / 6, 2 / 3, 0.25]))
basis = HCP.basis[0] + meta_basis
lattice = crystal.Crystal(HCP.lattice, basis[0:8], noreduce=True)
vacancyjumps = lattice.jumpnetwork(0, 1.01)
for pos, jlist in enumerate(vacancyjumps):
        if np.any([np.allclose(dx, [0.5, -0.8660254, 0.]) for (i, j), dx in jlist]):
            ind1 = pos
            break
#print("ind1 = ", ind1)
for pos, jlist in enumerate(vacancyjumps):
    if np.any([np.allclose(dx, [0.25, -0.4330127, 0.]) for (i, j), dx in jlist]):
        ind2 = pos
        break
#print("ind2 = ", ind2)
jumpnetwork = [vacancyjumps[1], vacancyjumps[ind2]]
jumpnetwork2 = (vacancyjumps[1], vacancyjumps[ind1])
meta_sites = tuple(np.arange(2, 8, 1))
chem = 0
starset = stars.StarSetMeta(jumpnetwork, lattice, chem, meta_sites=meta_sites)
starset.generate(2)
to_del = []
for state in starset.states:
    if state.i not in meta_sites and state.j in meta_sites:
        if np.allclose(0.866025403784,np.linalg.norm(state.dx)):
            to_del.append(state)

#jump_om2, jt, sp, ref, zerojumps, modjumps, rep_net = starset.jumpnetwork_omega2(jumpnetwork2)
#jump_om1, jt1, sp1, ref1, zerojumps1, modjumps1,rep_net1 = starset.jumpnetwork_omega1(to_del,jumpnetwork2)

jump_om2, jt, sp, ref, jt2 = starset.jumpnetwork_omega2(jumpnetwork2)
jump_om1, jt1, sp1, ref1,jt12  = starset.jumpnetwork_omega1(to_del,jumpnetwork2)

#vecstarset = stars.VectorStarSetMeta(starset=starset)
#rate0expand, rate0escape, rate2expand, rate2escape = vecstarset.rateexpansions(jump_om2, jt,omega2=True,refnetwork=ref, zero_jumps=zerojumps)
#bias0expand, bias2expand = self.vecstarset.biasexpansions(jumpnetwork_omega2, jt)
#for state in starset.states:
#    if state.i not in meta_sites and state.j in meta_sites:
#        print(state.dx, np.linalg.norm(state.dx))
