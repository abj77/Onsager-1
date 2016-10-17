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
        jumpnetwork1, jt, sp, ref, zerojumps, modjumps = self.starset.jumpnetwork_omega1()
        self.assertEqual(len(jumpnetwork1), 1)
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0, 0))
        self.assertEqual([], zerojumps)

        jumpnetwork2, jt, sp, ref, zerojumps, modjumps = self.starset.jumpnetwork_omega2()
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
        jumpnetwork, jt, sp, ref, zerojumps, modjumps = self.starset.jumpnetwork_omega1()
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
        jumpnetwork, jt, sp, ref, zerojumps, modjumps = self.starset.jumpnetwork_omega1()
        self.assertEqual(len(jumpnetwork), 4 + 1 + 2)
        self.assertEqual(len(ref), 4 + 1 + 2)
        self.assertEqual(sum(len(jlist) for jlist in jumpnetwork), 12 * 11 + 6 * 8 + 24 * 7 + 12 * 5)
        self.assertEqual(sum(len(jlist) for jlist in ref), 12 * 11 + 6 * 8 + 24 * 7 + 12 * 5)
        # check that nothing changed with the larger StarSet
        jumpnetwork2, jt, sp, ref, zerojumps, modjumps = self.starset.jumpnetwork_omega2()
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
            jumpnetwork, jt, sp, ref, zerojumps, modjumps = self.starset.jumpnetwork_omega1()
            for jumplist, (s1, s2) in zip(jumpnetwork, sp):
                for (i, f), dx in jumplist:
                    si = self.starset.index[i]
                    sf = self.starset.index[f]
                    self.assertTrue((s1, s2) == (si, sf) or (s1, s2) == (sf, si))


            for jumplist, w0list in zip(jumpnetwork, ref):
                for i, ((ji, jf), dx) in enumerate(jumplist):
                    self.assertTrue(np.allclose(dx, w0list[i].dx))


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
