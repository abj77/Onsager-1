"""
Temporary test file for the new meta classes in crystalStars. Will merge with test_crystalStars.
"""

__author__ = 'Abhinav Jain'

#

import unittest
import onsager.crystal as crystal
import numpy as np
import onsager.crystalStars as stars


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

class StarTests(unittest.TestCase):
    """Set of tests that our star code is behaving correctly for a general materials"""
    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.chem = 0
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites = self.meta_sites)

    def isclosed(self, starset, starindex):
        """Evaluate if star s is closed against group operations."""
        for i1 in starset.stars[starindex]:
            ps1 = starset.states[i1]
            for i2 in starset.stars[starindex]:
                ps2 = starset.states[i2]
                if not any(ps1 == ps2.g(self.crys, self.chem, g) for g in self.crys.G):
                    return False
        return True

    def testStarConsistent(self):
        """Check that the counts (Npts, Nstars) make sense, with Nshells = 1..4"""
        for n in range(1, 5):
            self.starset.generate(n)
            for starindex in range(self.starset.Nstars):
                self.assertTrue(self.isclosed(self.starset, starindex))

    def testStarindices(self):
        """Check that our indexing is correct."""
        self.starset.generate(4)
        for si, star in enumerate(self.starset.stars):
            for i in star:
                self.assertEqual(si, self.starset.starindex(self.starset.states[i]))
        self.assertEqual(None, self.starset.starindex(stars.PairState.zero()))
        self.assertEqual(None, self.starset.stateindex(stars.PairState.zero()))
        self.assertNotIn(stars.PairState.zero(), self.starset)  # test __contains__ (PS in starset)

    def assertEqualStars(self, s1, s2):
        """Asserts that two star sets are equal."""
        self.assertEqual(s1.Nstates, s2.Nstates,
                         msg='Number of states in two star sets are not equal: {} != {}'.format(
                             s1.Nstates, s2.Nstates))
        self.assertEqual(s1.Nshells, s2.Nshells,
                         msg='Number of shells in two star sets are not equal: {} != {}'.format(
                             s1.Nshells, s2.Nshells))
        self.assertEqual(s1.Nstars, s2.Nstars,
                         msg='Number of stars in two star sets are not equal: {} != {}'.format(
                             s1.Nstars, s2.Nstars))
        for s in s1.stars:
            # grab the first entry, and index it into a star; then check that all the others are there.
            ps0 = s1.states[s[0]]
            s2ind = s2.starindex(s1.states[s[0]])
            self.assertNotEqual(s2ind, -1,
                                msg='Could not find state {} from s1 in s2'.format(s1.states[s[0]]))
            self.assertEqual(len(s), len(s2.stars[s2ind]),
                             msg='Star in s1 has different length than star in s2? {} != {}'.format(
                                 len(s), len(s2.stars[s2ind])))
            for i1 in s:
                ps1 = s1.states[i1]
                self.assertEqual(s2ind, s2.starindex(ps1),
                                 msg='States {} and {} from star in s1 belong to different stars in s2'.format(
                                     ps0, ps1))

    def testStarCombine(self):
        """Check that we can combine two stars and get what we expect."""
        s1 = self.starset.copy()
        s2 = self.starset.copy()
        # s3 = self.starset.copy()
        s4 = self.starset.copy()
        s1.generate(1)
        s2.generate(1)
        s3 = s1 + s2
        s4.generate(2)
        # s3 = s1 + s2, should equal s4
        self.assertEqualStars(s1, s2)
        self.assertEqualStars(s3, s4)



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

    def testStarCount(self):
        """Check that the counts (Npts, Nstars) make sense for HCP meta, with Nshells = 1, 2, 3"""
        # 110
        self.starset.generate(1)
        self.assertEqual(self.starset.Nstars, 3)
        #self.assertEqual(self.starset.Nstates, 6)

        # 110, 200, 211, 220
        self.starset.generate(2)
        self.assertEqual(self.starset.Nstars, 9)

        # 110, 200, 211, 220, 310, 321, 330, 222
        #self.starset.generate(3)
        #self.assertEqual(self.starset.Nstars, 8)

class VectorStarTests(unittest.TestCase):
    """Set of tests that our VectorStar class is behaving correctly"""
    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.chem = 0
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

    def testVectorStarGenerate(self):
        """Can we generate star-vectors that make sense?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
        self.assertTrue(self.vecstarset.Nvstars > 0)

    def VectorStarConsistent(self, nshells):
        """Do the star vectors obey the definition?"""
        self.starset.generate(nshells)
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
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
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
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
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
        self.assertEqual(self.vecstarset.Nvstars, 3)

    def testVectorStarOuterProduct(self):
        """Do we generate the correct outer products for our star-vectors (symmetry checks)?"""
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
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

import onsager.GFcalc as GFcalc

class VectorStarGFlinearTests(unittest.TestCase):
    """Set of tests that make sure we can construct the GF matrix as a linear combination"""

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

        # not an accurate value for Nmax; but since we just need some values, it's fine
        self.GF = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.jumpnetwork, Nmax=2)
        self.GF.SetRates([1 for s in self.sitelist],
                         [0 for s in self.sitelist],
                         self.rates,
                         [0 for j in self.rates])

    def ConstructGF(self, nshells):
        self.starset.generate(nshells)
        self.vecstarset = stars.VectorStarSet(self.starset)
        GFexpand, GFstarset = self.vecstarset.GFexpansion()
        gexpand = np.zeros(GFstarset.Nstars)
        for i, star in enumerate(GFstarset.stars):
            st = GFstarset.states[star[0]]
            gexpand[i] = self.GF(st.i, st.j, st.dx)
        for i in range(self.vecstarset.Nvstars):
            for j in range(self.vecstarset.Nvstars):
                # test the construction
                # GFsum = np.sum(GFexpand[i,j,:])
                # if abs(GFsum) > 1e-5:
                #     print('GF vector star set between:')
                #     for R, v in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                #         print('  {} / {}'.format(self.starset.states[R], v))
                #     print('and')
                #     for R, v in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                #         print('  {} / {}'.format(self.starset.states[R], v))
                #     print('expansion:')
                #     for k, g in enumerate(GFexpand[i,j,:]):
                #         if abs(g) > 1e-5:
                #             print('  {:+0.15f}*{}'.format(g, GFstarset.states[k]))
                # self.assertAlmostEqual(GFsum, 0, msg='Failure for {},{}: GF= {}'.format(i,j,GFsum))
                g = 0
                for si, vi in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                    for sj, vj in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                        try:
                            ds = self.starset.states[sj] ^ self.starset.states[si]
                        except:
                            continue
                        g += np.dot(vi, vj) * self.GF(ds.i, ds.j, ds.dx)
                self.assertAlmostEqual(g, np.dot(GFexpand[i, j, :], gexpand))
                # Removed this test. It's not generally true.
                # self.assertAlmostEqual(np.sum(GFexpand), 0)
                # print(np.dot(GFexpand, gexpand))

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)

class VectorStarGFHCPlinearTests(VectorStarGFlinearTests):
    """Set of tests that make sure we can construct the GF matrix as a linear combination for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

        # not an accurate value for Nmax; but since we just need some values, it's fine
        self.GF = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.jumpnetwork, Nmax=2)
        self.GF.SetRates([1 for s in self.sitelist],
                         [0 for s in self.sitelist],
                         self.rates,
                         [0 for j in self.rates])

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)

class VectorStarOmega0Tests(unittest.TestCase):
    """Set of tests for our expansion of omega_0"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

    def testConstructOmega0(self):
        # NOTE: now we only take omega0 *here* to be those equivalent to omega1 jumps; the exchange
        # terms are handled in omega2; the jumps outside the kinetic shell simply contributed onsite escape
        # terms that get subtracted away, since the outer kinetic shell *has* to have zero energy
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        jumpnetwork_omega1, jt, sp, refnetwork_omega1, refjt = self.starset.jumpnetwork_omega1()
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
        rate0expand, rate0escape, rate1expand, rate1escape = self.vecstarset.rateexpansions(jumpnetwork_omega1, jt, refnetwork=refnetwork_omega1, jumptype2=refjt)
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


class VectorStarHCPOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)


class VectorStarOmegalinearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_1"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

    def testConstructOmega1(self):
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
        jumpnetwork_omega1, jt, sp, refnetwork_omega1, refjt = self.starset.jumpnetwork_omega1()
        rate0expand, rate0escape, rate1expand, rate1escape = self.vecstarset.rateexpansions(jumpnetwork_omega1, jt,
                                                                                            refnetwork=refnetwork_omega1,
                                                                                            jumptype2=refjt)
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


class VectorStarHCPOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)


class VectorStarOmega2linearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_2"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

    def testConstructOmega2(self):
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
        jumpnetwork_omega2, jt, sp, refnetwork_omega2, refjt = self.starset.jumpnetwork_omega2(jumpnetwork2=self.jumpnetwork2)
        rate0expand, rate0escape, rate2expand, rate2escape = self.vecstarset.rateexpansions(jumpnetwork_omega2, jt,
                                                                                            refnetwork=refnetwork_omega2,
                                                                                            jumptype2=refjt,omega2=True)
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


class VectorStarHCPOmega2linearTests(VectorStarOmega2linearTests):
    """Set of tests for our expansion of omega_2 for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

class VectorStarBias2linearTests(unittest.TestCase):
    """Set of tests for our expansion of bias vector (2)"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPMetarates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

    def testConstructBias2(self):
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
        jumpnetwork_omega2, jt, sp, refnetwork_omega2, refjt = self.starset.jumpnetwork_omega2(jumpnetwork2=self.jumpnetwork2)
        bias0expand, bias2expand = self.vecstarset.biasexpansions(jumpnetwork_omega2, jt,refnetwork=refnetwork_omega2,jumptype2=refjt,omega2=True)
        # make omega2 twice omega0:
        alpha = 2.
        om2expand = np.array([0.5,0.5]) #alpha * self.rates
        om0expand = np.array([0.5,1.0]) #self.rates.copy()
        self.assertEqual(np.shape(bias2expand),
                         (self.vecstarset.Nvstars, len(self.jumpnetwork)))
        biasvec1 = np.zeros((self.starset.Nstates, 3))  # bias vector: only the exchange hops
        biasvec2 = np.zeros((self.starset.Nstates, 3))  # bias vector: only the exchange hops

        for jumplist, rate in zip(jumpnetwork_omega2, om2expand):
            for (IS, FS), dx in jumplist:
                for i in range(self.starset.Nstates):
                    if IS == i:
                        biasvec1[i, :] += dx * rate

        for jumplist, rate in zip(refnetwork_omega2, om0expand):
            for (IS, FS), dx in jumplist:
                for i in range(self.starset.Nstates):
                    if IS == i:
                        biasvec2[i, :] += dx * rate

        biasvec = biasvec1 - biasvec2
        # construct the same bias vector using our expansion
        biasveccomp = np.zeros((self.starset.Nstates, 3))
        print(bias0expand)
        print(bias2expand)
        for b2, b0, svpos, svvec in zip(np.dot(bias2expand, om2expand),
                                        np.dot(bias0expand, om0expand),
                                        self.vecstarset.vecpos,
                                        self.vecstarset.vecvec):
            # test the construction
            for Ri, vi in zip(svpos, svvec):
                biasveccomp[Ri, :] += (b2 - b0) * vi

        #for i,j in zip (biasvec,biasveccomp):
        #    print(i,j)

        for i in range(self.starset.Nstates):
            self.assertTrue(np.allclose(biasvec[i], biasveccomp[i]),
                            msg='Failure for state {}: {}\n{} != {}'.format(
                                i, self.starset.states[i], biasvec[i], biasveccomp[i]))


class VectorStarHCPBias2linearTests(VectorStarBias2linearTests):
    """Set of tests for our expansion of bias vector (2) for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPMetarates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

class VectorStarBias1linearTests(unittest.TestCase):
    """Set of tests for our expansion of bias vector (1)"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPMetarates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

    def testConstructBias1(self):
        """Do we construct our omega1 bias correctly?"""
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
        jumpnetwork_omega1, jt, sp, refnetwork_omega1, refjt = self.starset.jumpnetwork_omega1()
        bias0expand, bias1expand = self.vecstarset.biasexpansions(jumpnetwork_omega1, jt,refnetwork=refnetwork_omega1,jumptype2=refjt)
        om1expand = np.random.uniform(0, 1, len(jumpnetwork_omega1))
        om0expand = self.rates.copy()
        self.assertEqual(np.shape(bias1expand),
                         (self.vecstarset.Nvstars, len(jumpnetwork_omega1)))
        biasvec = np.zeros((self.starset.Nstates, 3))  # bias vector: only the exchange hops

        for jumplist, rate, om0type in zip(jumpnetwork_omega1, om1expand, jt):
            om0 = om0expand[om0type]
            for (IS, FS), dx in jumplist:
                for i in range(self.starset.Nstates):
                    if IS == i:
                        biasvec[i, :] += dx * (rate - om0)
        # construct the same bias vector using our expansion
        biasveccomp = np.zeros((self.starset.Nstates, 3))
        for b1, b0, svpos, svvec in zip(np.dot(bias1expand, om1expand),
                                        np.dot(bias0expand, om0expand),
                                        self.vecstarset.vecpos,
                                        self.vecstarset.vecvec):
            # test the construction
            for Ri, vi in zip(svpos, svvec):
                biasveccomp[Ri, :] += (b1 - b0) * vi
        for i in range(self.starset.Nstates):
            self.assertTrue(np.allclose(biasvec[i], biasveccomp[i]),
                            msg='Failure for state {}: {}\n{} != {}'.format(
                                i, self.starset.states[i], biasvec[i], biasveccomp[i]))

    def testPeriodicBias(self):
        """Do we have no periodic bias?"""
        vectorbasislist = self.crys.FullVectorBasis(self.chem)[0]  # just check the VB list
        # we *should* have some projection if there's a vectorbasis, so only continue if this is empty
        if len(vectorbasislist) != 0: return
        self.starset.generate(1, originstates=True)  # turn on origin state generation
        self.vecstarset = stars.VectorStarSet(self.starset)
        for elemtype in ('solute', 'vacancy'):
            OSindices, folddown, OS_VB = self.vecstarset.originstateVectorBasisfolddown(elemtype)
            NOS = len(OSindices)
            self.assertEqual(NOS, 0)
            self.assertEqual(folddown.shape, (NOS, self.vecstarset.Nvstars))
            self.assertEqual(OS_VB.shape, (NOS, len(self.crys.basis[self.chem]), 3))


class VectorStarHCPBias1linearTests(VectorStarBias1linearTests):
    """Set of tests for our expansion of bias vector (1) for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPMetarates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

class VectorStarPeriodicBias(unittest.TestCase):
    """Set of tests for our expansion of periodic bias vector (1)"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.rates = HCPMetarates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSetMeta(self.jumpnetwork, self.crys, self.chem, meta_sites=self.meta_sites)

    def testOriginStates(self):
        """Does our origin state treatment work correctly to produce the vector bases?"""
        self.starset.generate(2, originstates=True)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSetMeta(self.starset)
        vectorbasislist = self.crys.FullVectorBasis(self.chem)[0]
        NVB = len(vectorbasislist)
        for elemtype, attr in zip(['vacancy', 'solute'], ['j', 'i']):
            OSindices, folddown, OS_VB = self.vecstarset.originstateVectorBasisfolddown(elemtype)
            NOS = len(OSindices)
            self.assertEqual(NOS, NVB)
            self.assertEqual(folddown.shape, (NOS, self.vecstarset.Nvstars))
            self.assertEqual(OS_VB.shape, (NOS,) + vectorbasislist[0].shape)
            for n, svR in enumerate(self.vecstarset.vecpos):
                if n in OSindices:
                    for i in svR:
                        self.assertTrue(self.starset.states[i].iszero())
                else:
                    for i in svR:
                        self.assertFalse(self.starset.states[i].iszero())
            for n in range(10):
                # test that our OS in our VectorStar make a proper basis according to our VB:
                vb = sum((2. * u - 1) * vect for u, vect in zip(np.random.random(len(vectorbasislist)), vectorbasislist))
                vb_proj = np.tensordot(OS_VB, np.tensordot(OS_VB, vb, axes=((1, 2), (0, 1))), axes=(0, 0))
                self.assertTrue(np.allclose(vb, vb_proj))
                # expand out to all sites:
                svexp = np.dot(folddown.T, np.tensordot(OS_VB, vb, axes=((1, 2), (0, 1))))
                vbdirect = np.array([vb[getattr(PS, attr)] for PS in self.starset.states])
                vbexp = np.zeros((self.starset.Nstates, 3))
                for svcoeff, svR, svv in zip(svexp, self.vecstarset.vecpos, self.vecstarset.vecvec):
                    for s, v in zip(svR, svv):
                        vbexp[s, :] += v * svcoeff
                self.assertTrue(np.allclose(vbdirect, vbexp))

