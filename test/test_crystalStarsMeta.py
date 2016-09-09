"""
Temporary test file for the new meta classes in crystalStars. Will merge with test_crystalStars.
"""

__author__ = 'Abhinav Jain'

#

import unittest
import onsager.crystal as crystal
import numpy as np
import onsager.crystalStars as stars


def setuportho():
    crys = crystal.Crystal(np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float),
                           [[np.zeros(3)]])
    jumpnetwork = [[((0, 0), np.array([3., 0., 0.])), ((0, 0), np.array([-3., 0., 0.]))],
                   [((0, 0), np.array([0., 2., 0.])), ((0, 0), np.array([0., -2., 0.]))],
                   [((0, 0), np.array([0., 0., 1.])), ((0, 0), np.array([0., 0., -1.]))]]
    return crys, jumpnetwork


def orthorates():
    return np.array([3., 2., 1.])

def setupcubic():
    crys = crystal.Crystal(np.eye(3), [[np.zeros(3)]])
    jumpnetwork = [[((0, 0), np.array([1., 0., 0.])), ((0, 0), np.array([-1., 0., 0.])),
                    ((0, 0), np.array([0., 1., 0.])), ((0, 0), np.array([0., -1., 0.])),
                    ((0, 0), np.array([0., 0., 1.])), ((0, 0), np.array([0., 0., -1.]))]]
    return crys, jumpnetwork


def cubicrates():
    return np.array([1. / 6.])


def setupFCC():
    lattice = crystal.Crystal.FCC(2.)
    jumpnetwork = lattice.jumpnetwork(0, 2. * np.sqrt(0.5) + 0.01)
    return lattice, jumpnetwork


def FCCrates():
    return np.array([1. / 12.])


def setupHCP():
    lattice = crystal.Crystal.HCP(1.)
    jumpnetwork = lattice.jumpnetwork(0, 1.01)
    return lattice, jumpnetwork

def setupHCPMeta():
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
    jumpnetwork2 = [vacancyjumps[1], vacancyjumps[ind1]]
    meta_sites = np.arange(2, 8, 1)
    return lattice, jumpnetwork, jumpnetwork2, meta_sites


def HCPrates():
    return np.array([1. / 12., 1. / 12.])

def HCPMetarates():
    return np.array([1. / 12., 1. / 24.])


# replaced DoubleStarTests
class JumpNetworkTests(unittest.TestCase):
    """Set of tests that our JumpNetwork is behaving correctly."""
    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.chem = 0
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem)

    def testJumpNetworkGeneration(self):
        """Can we generate jumpnetworks?"""
        self.starset.generate(1)
        jumpnetwork1, jt, sp, ref, zerojumps = self.starset.jumpnetwork_omega1()
        self.assertEqual(len(jumpnetwork1), 1)
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0, 0))
        self.assertEqual([], zerojumps)

        jumpnetwork2, jt, sp, ref, zerojumps = self.starset.jumpnetwork_omega2()
        self.assertEqual(len(jumpnetwork2), 1)
        self.assertEqual(len(jumpnetwork2[0]), len(self.jumpnetwork[0]))
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0, 0))


    def testJumpNetworkCount(self):
        """Check that the counts in the jumpnetwork make sense for FCC, with Nshells = 1, 2"""
        # each of the 12 <110> pairs to 101, 10-1, 011, 01-1 = 4, so should be 48 pairs
        # (which includes "double counting": i->j and j->i)
        # but *all* of those 48 are all equivalent to each other by symmetry: one jump network.
        self.starset.generate(1)
        jumpnetwork, jt, sp, ref, zerojumps = self.starset.jumpnetwork_omega1()
        self.assertEqual(len(jumpnetwork), 1)
        self.assertEqual(len(ref), 1)
        self.assertEqual(len(jumpnetwork[0]), 48)
        self.assertEqual(len(ref[0]), 48)
        # Now have four stars (110, 200, 211, 220), so this means
        # 12 <110> pairs to 11 (no 000!); 12*11
        # 6 <200> pairs to 110, 101, 1-10, 10-1; 211, 21-1, 2-11, 2-1-1 = 8; 6*8
        # 24 <211> pairs to 110, 101; 200; 112, 121; 202, 220 = 7; 24*7
        # 12 <220> pairs to 110; 12-1, 121, 21-1, 211 = 5; 12*5
        # unique pairs: (110, 101); (110, 200); (110, 211); (110, 220); (200, 211); (211, 112); (211, 220)
        self.starset.generate(2)
        self.assertEqual(self.starset.Nstars, 4)
        jumpnetwork, jt, sp, ref, zerojumps = self.starset.jumpnetwork_omega1()
        self.assertEqual(len(jumpnetwork), 4 + 1 + 2)
        self.assertEqual(len(ref), 4 + 1 + 2)
        self.assertEqual(sum(len(jlist) for jlist in jumpnetwork), 12 * 11 + 6 * 8 + 24 * 7 + 12 * 5)
        self.assertEqual(sum(len(jlist) for jlist in ref), 12 * 11 + 6 * 8 + 24 * 7 + 12 * 5)
        # check that nothing changed with the larger StarSet
        jumpnetwork2, jt, sp, ref, zerojumps = self.starset.jumpnetwork_omega2()
        self.assertEqual(len(jumpnetwork2), 1)
        self.assertEqual(len(ref), 1)
        self.assertEqual(len(jumpnetwork2[0]), len(self.jumpnetwork[0]))
        self.assertEqual(len(ref[0]), len(self.jumpnetwork[0]))
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0, 0))

    def testJumpNetworkindices(self):
        """Check that our indexing works correctly for Nshell=1..3"""
        for nshells in range(1, 4):
            self.starset.generate(nshells)
            jumpnetwork, jt, sp, ref, zerojumps = self.starset.jumpnetwork_omega1()
            for jumplist, (s1, s2) in zip(jumpnetwork, sp):
                for (i, f), dx in jumplist:
                    si = self.starset.index[i]
                    sf = self.starset.index[f]
                    self.assertTrue((s1, s2) == (si, sf) or (s1, s2) == (sf, si))


            for jumplist, w0list in zip(jumpnetwork, ref):
                for i, ((ji, jf), dx) in enumerate(jumplist):
                    self.assertTrue(np.allclose(dx, w0list[i].dx))

class VectorStarTests(unittest.TestCase):
    """Set of tests that our VectorStar class is behaving correctly"""
    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarGenerate(self):
        """Can we generate star-vectors that make sense?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        self.assertTrue(self.vecstarset.Nvstars > 0)

    def VectorStarConsistent(self, nshells):
        """Do the star vectors obey the definition?"""
        self.starset.generate(nshells)
        self.vecstarset = stars.VectorStarSet(self.starset)
        for s, vec in zip(self.vecstarset.vecpos, self.vecstarset.vecvec):
            for si, v in zip(s, vec):
                PS = self.starset.states[si]
                for g in self.crys.G:
                    gsi = self.starset.stateindex(PS.g(self.crys, self.chem, g))
                    vrot = self.crys.g_direc(g, v)
                    for si1, v1 in zip(s, vec):
                        if gsi == si1: self.assertTrue(np.allclose(v1, vrot))

    def VectorStarOrthonormal(self, nshells):
        """Are the star vectors orthonormal?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        for s1, v1 in zip(self.vecstarset.vecpos, self.vecstarset.vecvec):
            for s2, v2 in zip(self.vecstarset.vecpos, self.vecstarset.vecvec):
                if s1[0] == s2[0]:
                    dp = sum(np.dot(vv1, vv2) for vv1, vv2 in zip(v1, v2))
                    if np.allclose(v1[0], v2[0]):
                        self.assertAlmostEqual(1., dp,
                                               msg='Failed normality for {}/{} and {}/{}'.format(
                                                   self.starset.states[s1[0]], v1[0],
                                                   self.starset.states[s2[0]], v2[0]))
                    else:
                        self.assertAlmostEqual(0., dp,
                                               msg='Failed orthogonality for {}/{} and {}/{}'.format(
                                                   self.starset.states[s1[0]], v1[0],
                                                   self.starset.states[s2[0]], v2[0]))

    def testVectorStarConsistent(self):
        """Do the star vectors obey the definition?"""
        self.VectorStarConsistent(2)

    def testVectorStarOrthonormal(self):
        self.VectorStarOrthonormal(2)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        self.assertEqual(self.vecstarset.Nvstars, 3)

    def testVectorStarOuterProduct(self):
        """Do we generate the correct outer products for our star-vectors (symmetry checks)?"""
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSet(self.starset)
        self.assertEqual(np.shape(self.vecstarset.outer),
                         (3, 3, self.vecstarset.Nvstars, self.vecstarset.Nvstars))
        # check our diagonal blocks first:
        for outer in [self.vecstarset.outer[:, :, i, i]
                      for i in range(self.vecstarset.Nvstars)]:
            self.assertAlmostEqual(np.trace(outer), 1)
            # should also be symmetric:
            for g in self.crys.G:
                g_out_gT = self.crys.g_tensor(g, outer)
                self.assertTrue(np.allclose(g_out_gT, outer))
        # off-diagonal terms now
        for outer in [self.vecstarset.outer[:, :, i, j]
                      for i in range(self.vecstarset.Nvstars)
                      for j in range(self.vecstarset.Nvstars)
                      if i != j]:
            self.assertAlmostEqual(np.trace(outer), 0)
            # should also be symmetric:
            for g in self.crys.G:
                g_out_gT = self.crys.g_tensor(g, outer)
                self.assertTrue(np.allclose(g_out_gT, outer))
        for i, (s0, svv0) in enumerate(zip(self.vecstarset.vecpos, self.vecstarset.vecvec)):
            for j, (s1, svv1) in enumerate(zip(self.vecstarset.vecpos, self.vecstarset.vecvec)):
                testouter = np.zeros((3, 3))
                if s0[0] == s1[0]:
                    # we have the same underlying star to work with, so...
                    for v0, v1 in zip(svv0, svv1):
                        testouter += np.outer(v0, v1)
                self.assertTrue(np.allclose(self.vecstarset.outer[:, :, i, j], testouter),
                                msg='Failed for vector stars {} and {}:\n{} !=\n{}'.format(
                                    i, j, self.vecstarset.outer[:, :, i, j], testouter))


class VectorStarFCCTests(VectorStarTests):
    """Set of tests that our VectorStar class is behaving correctly, for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # nn + nn = 4 stars, and that should make 5 star-vectors!
        self.assertEqual(self.starset.Nstars, 4)
        self.assertEqual(self.vecstarset.Nvstars, 5)

    def testVectorStarConsistent(self):
        """Do the star vectors obey the definition?"""
        self.VectorStarConsistent(2)

    def testVectorStarOuterProductMore(self):
        """Do we generate the correct outer products for our star-vectors?"""
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # with cubic symmetry, these all have to equal 1/3 * identity, and
        # with a diagonal matrix
        testouter = 1. / 3. * np.eye(3)
        for outer in [self.vecstarset.outer[:, :, i, i]
                      for i in range(self.vecstarset.Nvstars)]:
            self.assertTrue(np.allclose(outer, testouter))
        for outer in [self.vecstarset.outer[:, :, i, j]
                      for i in range(self.vecstarset.Nvstars)
                      for j in range(self.vecstarset.Nvstars)
                      if i != j]:
            self.assertTrue(np.allclose(outer, 0))


class VectorStarHCPTests(VectorStarTests):
    """Set of tests that our VectorStar class is behaving correctly, for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # two stars, with two vectors: one basal, one along c (more or less)
        self.assertEqual(self.starset.Nstars, 2)
        self.assertEqual(self.vecstarset.Nvstars, 2 + 2)


class VectorStarOmega0Tests(unittest.TestCase):
    """Set of tests for our expansion of omega_0"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructOmega0(self):
        # NOTE: now we only take omega0 *here* to be those equivalent to omega1 jumps; the exchange
        # terms are handled in omega2; the jumps outside the kinetic shell simply contributed onsite escape
        # terms that get subtracted away, since the outer kinetic shell *has* to have zero energy
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        jumpnetwork_omega1, jt, sp = self.starset.jumpnetwork_omega1()
        self.vecstarset = stars.VectorStarSet(self.starset)
        rate0expand, rate0escape, rate1expand, rate1escape = self.vecstarset.rateexpansions(jumpnetwork_omega1, jt)
        # rate0expand = self.vecstarset.rate0expansion()
        self.assertEqual(np.shape(rate0expand),
                         (self.vecstarset.Nvstars, self.vecstarset.Nvstars, len(self.jumpnetwork)))
        om0expand = self.rates.copy()
        # put together the onsite and off-diagonal terms for our matrix:
        # go through each state, and add the escapes for the vacancy; see if vacancy (PS.j)
        # is the initial state (i) for a transition out (i,f), dx
        om0matrix = np.zeros((self.starset.Nstates, self.starset.Nstates))
        for ns, PS in enumerate(self.starset.states):
            for rate, jumplist in zip(self.rates, self.starset.jumpnetwork_index):
                for TS in [self.starset.jumplist[jumpindex] for jumpindex in jumplist]:
                    if PS.j == TS.i:
                        nsend = self.starset.stateindex(PS + TS)
                        if nsend is not None:
                            om0matrix[ns, nsend] += rate
                            om0matrix[ns, ns] -= rate
        # now, we need to convert that omega0 matrix into the "folded down"
        for i, (sRv0, svv0) in enumerate(zip(self.vecstarset.vecpos, self.vecstarset.vecvec)):
            for j, (sRv1, svv1) in enumerate(zip(self.vecstarset.vecpos, self.vecstarset.vecvec)):
                om0_sv = 0
                for i0, v0 in zip(sRv0, svv0):
                    for i1, v1 in zip(sRv1, svv1):
                        om0_sv += np.dot(v0, v1) * om0matrix[i0, i1]
                om0_sv_comp = np.dot(rate0expand[i, j], om0expand)
                if i == j: om0_sv_comp += np.dot(rate0escape[i], om0expand)
                self.assertAlmostEqual(om0_sv, om0_sv_comp,
                                       msg='Failed to match {}, {}: {} != {}'.format(
                                           i, j, om0_sv, om0_sv_comp))


class VectorStarFCCOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarHCPOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)



class HCPMetaTests(unittest.TestCase):
    """Set of tests specific to the HCP lattice with metastable states."""
    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.chem = 0
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

    def testStateGeneration(self):
        """Check that the generate code is working correctly for metastable states"""
        self.starset.generate(2)
        print(self.starset.states)
        for s in self.starset.states:
            self.assertTrue(s.i not in self.meta_sites)

class VectorStarOmegalinearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_1"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        # self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructOmega1(self):
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSet(self.starset)
        jumpnetwork_omega1, jt, sp = self.starset.jumpnetwork_omega1()
        rate0expand, rate0escape, rate1expand, rate1escape = self.vecstarset.rateexpansions(jumpnetwork_omega1, jt)
        # rate1expand = self.vecstarset.rate1expansion(jumpnetwork)
        self.assertEqual(np.shape(rate1expand),
                         (self.vecstarset.Nvstars, self.vecstarset.Nvstars, len(jumpnetwork_omega1)))
        # make some random rates
        om1expand = np.random.uniform(0, 1, len(jumpnetwork_omega1))
        for i in range(self.vecstarset.Nvstars):
            for j in range(self.vecstarset.Nvstars):
                # test the construction
                om1 = 0
                for Ri, vi in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                    for Rj, vj in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                        for jumplist, rate in zip(jumpnetwork_omega1, om1expand):
                            for (IS, FS), dx in jumplist:
                                if IS == Ri:
                                    if IS == Rj: om1 -= np.dot(vi, vj) * rate  # onsite terms...
                                    if FS == Rj: om1 += np.dot(vi, vj) * rate
                om1_sv_comp = np.dot(rate1expand[i, j], om1expand)
                if i == j: om1_sv_comp += np.dot(rate1escape[i], om1expand)
                self.assertAlmostEqual(om1, om1_sv_comp,
                                       msg='Failed to match {}, {}: {} != {}'.format(
                                           i, j, om1, om1_sv_comp))
                # print(np.dot(rateexpand, om1expand))


class VectorStarFCCOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        # self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

class VectorStarHCPOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        # self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

class VectorStarOmega2linearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_2"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructOmega2(self):
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSet(self.starset)
        jumpnetwork_omega2, jt, sp = self.starset.jumpnetwork_omega2()
        rate0expand, rate0escape, rate2expand, rate2escape = self.vecstarset.rateexpansions(jumpnetwork_omega2, jt)

        # construct the set of rates corresponding to the unique stars:
        om2expand = self.rates.copy()
        self.assertEqual(np.shape(rate2expand),
                         (self.vecstarset.Nvstars, self.vecstarset.Nvstars, len(jumpnetwork_omega2)))
        for i in range(self.vecstarset.Nvstars):
            for j in range(self.vecstarset.Nvstars):
                # test the construction
                om2 = 0
                for Ri, vi in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                    for Rj, vj in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                        for jumplist, rate in zip(jumpnetwork_omega2, om2expand):
                            for (IS, FS), dx in jumplist:
                                if IS == Ri:
                                    if IS == Rj: om2 -= np.dot(vi, vj) * rate  # onsite terms...
                                    if FS == Rj: om2 += np.dot(vi, vj) * rate
                om2_sv_comp = np.dot(rate2expand[i, j], om2expand)
                if i == j: om2_sv_comp += np.dot(rate2escape[i], om2expand)
                self.assertAlmostEqual(om2, om2_sv_comp,
                                       msg='Failed to match {}, {}: {} != {}'.format(
                                           i, j, om2, om2_sv_comp))
