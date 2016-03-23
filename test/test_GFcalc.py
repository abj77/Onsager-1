"""
Unit tests for calculation of lattice Green function for diffusion
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
from scipy import special
import onsager.GFcalc as GFcalc
import onsager.crystal as crystal

def poleFT(di, u, pm, erfupm=-1):
    """
    Calculates the pole FT (excluding the volume prefactor) given the `di` eigenvalues,
    the value of u magnitude (available from unorm), and the pmax scaling factor.

    :param di : array [:]  eigenvalues of `D2`
    :param u : double  magnitude of u, from unorm() = x.D^-1.x
    :param pm : double  scaling factor pmax for exponential cutoff function
    :param erfupm : double, optional  value of erf(0.5*u*pm) (negative = not set, then its calculated)
    :return poleFT : double
        integral of Gaussian cutoff function corresponding to a l=0 pole;
        erf(0.5*u*pm)/(4*pi*u*sqrt(d1*d2*d3)) if u>0
        pm/(4*pi^3/2 * sqrt(d1*d2*d3)) if u==0
    """

    if (u == 0):
        return 0.25 * pm / np.sqrt(np.product(di * np.pi))
    if (erfupm < 0):
        erfupm = special.erf(0.5 * u * pm)
    return erfupm * 0.25 / (np.pi * u * np.sqrt(np.product(di)))


class GreenFuncCrystalTests(unittest.TestCase):
    """Test new implementation of GF calculator, based on Crystal class"""
    def setUp(self):
        pass

    def testFCC(self):
        """Test on FCC"""
        FCC = crystal.Crystal.FCC(1.)
        FCC_sitelist = FCC.sitelist(0)
        FCC_jumpnetwork = FCC.jumpnetwork(0, 0.75)
        FCC_GF = GFcalc.GFCrystalcalc(FCC, 0, FCC_sitelist, FCC_jumpnetwork, Nmax=4)
        FCC_GF.SetRates([1],[0],[1],[0])
        # test the pole function:
        for u in np.linspace(0,5,21):
            pole_orig = FCC_GF.crys.volume*poleFT(FCC_GF.d, u, FCC_GF.pmax)
            pole_new = FCC_GF.g_Taylor_fnlu[(-2,0)](u).real
            self.assertAlmostEqual(pole_orig, pole_new, places=15, msg="Pole (-2,0) failed for u={}".format(u))
        # test the discontinuity function:
        for u in np.linspace(0,5,21):
            disc_orig = FCC_GF.crys.volume*(FCC_GF.pmax/(2*np.sqrt(np.pi)))**3*\
                        np.exp(-(0.5*u*FCC_GF.pmax)**2)/np.sqrt(np.product(FCC_GF.d))
            disc_new = FCC_GF.g_Taylor_fnlu[(0,0)](u).real
            self.assertAlmostEqual(disc_orig, disc_new, places=15, msg="Disc (0,0) failed for u={}".format(u))
        # test the GF evaluation against the original
        # NNvect = np.array([dx for (i,j), dx in FCC_jumpnetwork[0]])
        # rates = np.array([1 for jump in NNvect])
        # old_FCC_GF = GFcalc.GFcalc(self.FCC.lattice, NNvect, rates)
        # for R in [np.array([0.,0.,0.]), np.array([0.5, 0.5, 0.]), np.array([0.5, 0., 0.5]), \
        #          np.array([1.,0.,0.]), np.array([1.,0.5,0.5]), np.array([1.,1.,0.])]:
        #     GF_orig = old_FCC_GF.GF(R)
        #     GF_new = FCC_GF(0,0,R)
        #     # print("R={}: dG= {}  G_orig= {}  G_new= {}".format(R, GF_new-GF_orig, GF_orig, GF_new))
        #     self.assertAlmostEqual(GF_orig, GF_new, places=5,
        #                            msg="Failed for R={}".format(R))

    def testHCP(self):
        """Test on HCP"""
        HCP = crystal.Crystal.HCP(1., np.sqrt(8/3))
        HCP_sitelist = HCP.sitelist(0)
        HCP_jumpnetwork = HCP.jumpnetwork(0, 1.01)
        HCP_GF = GFcalc.GFCrystalcalc(HCP, 0, HCP_sitelist, HCP_jumpnetwork, Nmax=4)
        HCP_GF.SetRates([1],[0],[1,1],[0,0])  # one unique site, two types of jumps
        # print(HCP_GF.Diffusivity())
        # make some basic vectors:
        hcp_basal = HCP.pos2cart(np.array([1.,0.,0.]), (0,0)) - \
                    HCP.pos2cart(np.array([0.,0.,0.]), (0,0))
        hcp_pyram = HCP.pos2cart(np.array([0.,0.,0.]), (0,1)) - \
                    HCP.pos2cart(np.array([0.,0.,0.]), (0,0))
        hcp_zero = np.zeros(3)
        for R in [hcp_zero, hcp_basal, hcp_pyram]:
            self.assertAlmostEqual(HCP_GF(0,0,R), HCP_GF(1,1,R), places=15)
        self.assertAlmostEqual(HCP_GF(0,0,hcp_basal), HCP_GF(0,0,-hcp_basal), places=15)
        self.assertAlmostEqual(HCP_GF(0,1,hcp_pyram), HCP_GF(1,0,-hcp_pyram), places=15)
        g0 = HCP_GF(0,0,hcp_zero)
        gbasal = HCP_GF(0,0,hcp_basal)
        gpyram = HCP_GF(0,1,hcp_pyram)
        self.assertAlmostEqual(-12*g0 + 6*gbasal + 6*gpyram, 1, places=6)
        # Try again, but with different rates:
        HCP_GF.SetRates([1],[0],[1,3],[0,0])  # one unique site, two types of jumps
        g0 = HCP_GF(0,0,hcp_zero)
        gw = 0
        for jumplist, omega in zip(HCP_jumpnetwork, HCP_GF.symmrate*HCP_GF.maxrate):
            for (i,j), dx in jumplist:
                if (i==0):
                    gw += omega*(HCP_GF(i,j,dx) - g0)
        self.assertAlmostEqual(gw, 1, places=6)

    def testBCC_B2(self):
        """Test that BCC and B2 produce the same GF"""
        a0 = 1.
        chem = 0
        BCC = crystal.Crystal.BCC(a0)
        BCC_sitelist = BCC.sitelist(chem)
        BCC_jumpnetwork = BCC.jumpnetwork(chem, 0.87*a0)
        BCC_GF = GFcalc.GFCrystalcalc(BCC, chem, BCC_sitelist, BCC_jumpnetwork, Nmax=6)
        BCC_GF.SetRates(np.ones(len(BCC_sitelist)),np.zeros(len(BCC_sitelist)),
                        2.*np.ones(len(BCC_jumpnetwork)), np.zeros(len(BCC_jumpnetwork)))

        B2 = crystal.Crystal(a0*np.eye(3), [np.zeros(3), np.array([0.45, 0.45, 0.45])])
        B2_sitelist = B2.sitelist(chem)
        B2_jumpnetwork = B2.jumpnetwork(chem, 0.99*a0)
        B2_GF = GFcalc.GFCrystalcalc(B2, chem, B2_sitelist, B2_jumpnetwork, Nmax=6)
        B2_GF.SetRates(np.ones(len(B2_sitelist)),np.zeros(len(B2_sitelist)),
                        2.*np.ones(len(B2_jumpnetwork)), np.zeros(len(B2_jumpnetwork)))
        veclist = [np.array([a0, 0, 0]), np.array([0, a0, 0]), np.array([0, 0, a0]),
                   np.array([-a0, 0, 0]), np.array([0, -a0, 0]), np.array([0, 0, -a0])]
        for v1 in veclist:
            for v2 in veclist:
                # print('{}: '.format(v1+v2) + '{} vs {} vs {}'.format(B2_GF(0,0,v1+v2),B2_GF(1,1,v1+v2),BCC_GF(0,0,v1+v2)))
                self.assertAlmostEqual(BCC_GF(0,0,v1+v2), B2_GF(0,0,v1+v2), places=5)
                self.assertAlmostEqual(BCC_GF(0, 0, v1 + v2), B2_GF(1, 1, v1 + v2), places=5)
        for jlist in B2_jumpnetwork:
            for (i,j), dx in jlist:
                # convert our B2 dx into a corresponding BCC dx:
                BCCdx = (0.5*a0)*np.round(dx/(0.5*a0))
                # print('({},{}), {} / {}: '.format(i,j,dx,BCCdx) + '{} vs {}'.format(B2_GF(i,j,dx), BCC_GF(0,0,BCCdx)))
                self.assertAlmostEqual(BCC_GF(0, 0, BCCdx), B2_GF(i, j, dx), places=5)

