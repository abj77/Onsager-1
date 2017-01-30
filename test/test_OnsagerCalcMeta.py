"""
Unit tests for Onsager calculator for vacancy-mediated diffusion.
"""

__author__ = 'Dallas R. Trinkle'


import unittest
import textwrap, itertools, types
import logging, inspect
import numpy as np
import onsager.OnsagerCalc as OnsagerCalc
import onsager.crystal as crystal

# uncomment for verbosity:
# logging.basicConfig(level=logging.DEBUG)  # VERBOSE


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


class DiffusionTestCase(unittest.TestCase):
    """Base class to define some diffusion-based assertions--contains no tests"""

    longMessage = False

    def makeunitythermodict(self, diffuser, solutebinding=1.):
        """Return a thermo dictionary with probability 1 for everything--or a solutebinding factor"""
        tdict = {'preV': np.ones(len(diffuser.sitelist)), 'eneV': np.zeros(len(diffuser.sitelist)),
                 'preS': np.ones(len(diffuser.sitelist)), 'eneS': np.zeros(len(diffuser.sitelist)),
                 'preT0': np.ones(len(diffuser.om0_jn)), 'eneT0': np.zeros(len(diffuser.om0_jn)),
                 'preSV': solutebinding * np.ones(len(diffuser.interactlist())),
                 'eneSV': np.zeros(len(diffuser.interactlist()))}
        tdict.update(diffuser.makeLIMBpreene(**tdict))
        return tdict

    def assertOrderingSuperEqual(self, s0, s1, msg=""):
        if s0 != s1:
            failmsg = msg + '\n'
            for line0, line1 in itertools.zip_longest(s0.__str__().splitlines(),
                                                      s1.__str__().splitlines(),
                                                      fillvalue=' - '):
                failmsg += line0 + '\t' + line1 + '\n'
            self.fail(msg=failmsg)

    # we use MappingProxyType to make a frozen dictionary:
    def assertEqualDiffusivity(self, diffuser1, tdict1, diffuser2, tdict2, msg="", kTlist=(1.,),
                               diffuserargs1=types.MappingProxyType({}),
                               diffuserargs2=types.MappingProxyType({})):
        """Assert that two diffusers give equal values over the same kT set"""
        for kT in kTlist:
            Lvv1, Lss1, Lsv1, L1vv1 = diffuser1.Lij(*diffuser1.preene2betafree(kT, **tdict1), **diffuserargs1)
            Lvv2, Lss2, Lsv2, L1vv2 = diffuser2.Lij(*diffuser2.preene2betafree(kT, **tdict2), **diffuserargs2)
            if hasattr(self, 'logger') and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug('kT={}'.format(kT))
                self.logger.debug('\n{}\n{}'.format(diffuser1, diffuserargs1))
                for Lname in ('Lvv1', 'Lss1', 'Lsv1', 'L1vv1'):
                    self.logger.debug('{}:\n{}'.format(Lname, locals()[Lname]))
                self.logger.debug('\n{}\n{}'.format(diffuser2, diffuserargs2))
                for Lname in ('Lvv2', 'Lss2', 'Lsv2', 'L1vv2'):
                    self.logger.debug('{}:\n{}'.format(Lname, locals()[Lname]))
            failmsg = ''
            for L, Lp, Lname in zip([Lvv1, Lss1, Lsv1, L1vv1],
                                    [Lvv2, Lss2, Lsv2, L1vv2],
                                    ['Lvv', 'Lss', 'Lsv', 'L1vv2']):
                if not np.allclose(L, Lp, atol=1e-7):
                    failmsg += textwrap.dedent("""\
                    Diffusivity {} does not match at kT={}?
                    {}
                    !=
                    {}
                    """).format(Lname, kT, L, Lp)
        if failmsg != '':
            self.fail(msg=textwrap.dedent("""\
            {}
            D1args={}, D2args={}
            {}""").format(msg, diffuserargs1, diffuserargs2, failmsg))


class CrystalOnsagerTestsHCP(DiffusionTestCase):
    """Test our new crystal-based vacancy-mediated diffusion calculator"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork, self.jumpnetwork2, self.meta_sites = setupHCPMeta()
        self.chem = 0
        self.a0 = 1.
        # self.jumpnetwork = self.crys.jumpnetwork(self.chem, 1.01 * self.a0)
        self.sitelist = self.crys.sitelist(self.chem)
        self.crystalname = 'Hexagonal Closed-Packed a0={} c0=sqrt(8/3)'.format(self.a0)
        # Correlation factors from doi://10.1080/01418617808239187
        # S. Ishioka and M. Koiwa, Phil. Mag. A 37, 517-533 (1978)
        # which they say matches older results in K. Compaan and C. Haven,
        # Trans. Faraday Soc. 52, 786 (1958) and ibid. 54, 1498
        self.correlx = 0.78120489
        self.correlz = 0.78145142

    def testmetatracer(self):
        """Test that HCP meta tracer works as expected"""
        self.logger = logging.getLogger(__name__ + '.' +
                                        self.__class__.__name__ + '.' +
                                        inspect.currentframe().f_code.co_name)
        # Make a calculator with one neighbor shell
        kT = 1.
        Diffusivity = OnsagerCalc.VacancyMediatedMeta(self.crys, self.chem, self.sitelist, self.jumpnetwork, 2,
                                                      meta_sites = self.meta_sites, jumpnetwork2= self.jumpnetwork2)

        HCPtracer = {'preV': np.array([1.0, 1.0 / 10000000.0]), 'eneV': np.array([0.0, 0.0]),
                     'preT0': np.array([0.5, 1.0]),
                     'eneT0': np.array([0, 0]),
                     }
        HCPtracer.update(Diffusivity.maketracerpreene(**HCPtracer))
        HCPtracer['preT2'] = np.array([0.5, 0.5])
        # thermaldef = self.makeunitythermodict(Diffusivity)
        L0vv = np.zeros((3, 3))
        # om0 = thermaldef['preT0'][0] / thermaldef['preV'][0] * \
        #       np.exp((thermaldef['eneV'][0] - thermaldef['eneT0'][0]) / kT)
        # for jumplist in self.jumpnetwork:
        #    for (i, j), dx in jumplist:
        #        L0vv += 0.5 * np.outer(dx, dx) * om0
        # L0vv /= self.crys.N

        Lvv, Lss, Lsv, L1vv = Diffusivity.Lij(*Diffusivity.preene2betafree(kT, **HCPtracer))

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('Crystal: {}\n{}'.format(self.crystalname, Diffusivity))
            for Lname in ('Lvv', 'Lss', 'Lsv', 'L1vv'):
                self.logger.debug('{}:\n{}'.format(Lname, locals()[Lname]))
        # we leave out Lss since it is not, in fact, isotropic!
        for L in [Lvv, Lsv, L1vv]:
            self.assertTrue(np.allclose(L, L[0, 0] * np.eye(3), atol=1e-6),
                            msg='Diffusivity not isotropic?')
        # No solute drag, so Lsv = -Lvv; Lvv = normal vacancy diffusion
        # all correlation is in that geometric prefactor of Lss.
        #self.assertTrue(np.allclose(Lvv, L0vv))
        #self.assertTrue(np.allclose(-Lsv, L0vv))
        self.assertTrue(np.allclose(L1vv, 0.))
        correlmat = np.array([[self.correlx, 0, 0], [0, self.correlx, 0], [0, 0, self.correlz]])
        self.assertTrue(np.allclose(-Lss, np.dot(correlmat, Lsv), rtol=1e-6),
                        msg='Failure to match correlation ({}, {}), got {}, {}'.format(
                            self.correlx, self.correlz, -Lss[0, 0] / Lsv[0, 0], -Lss[2, 2] / Lsv[2, 2]))
        # test large_om2 version:
        #self.assertEqualDiffusivity(Diffusivity, thermaldef, Diffusivity, thermaldef,
        #                            diffuserargs2={'large_om2': 0}, msg='large omega test fail')


    # def testtracer(self):
    #     """Test that HCP tracer works as expected"""
    #     self.logger = logging.getLogger(__name__ + '.' +
    #                                     self.__class__.__name__ + '.' +
    #                                     inspect.currentframe().f_code.co_name)
    #     # Make a calculator with one neighbor shell
    #     kT = 1.
    #     Diffusivity = OnsagerCalc.VacancyMediatedMeta(self.crys, self.chem, self.sitelist, self.jumpnetwork, 2,
    #                                                   meta_sites = self.meta_sites, jumpnetwork2= self.jumpnetwork2)
    #     thermaldef = self.makeunitythermodict(Diffusivity)
    #     L0vv = np.zeros((3, 3))
    #     om0 = thermaldef['preT0'][0] / thermaldef['preV'][0] * \
    #           np.exp((thermaldef['eneV'][0] - thermaldef['eneT0'][0]) / kT)
    #     for jumplist in self.jumpnetwork:
    #         for (i, j), dx in jumplist:
    #             L0vv += 0.5 * np.outer(dx, dx) * om0
    #     L0vv /= self.crys.N
    #     Lvv, Lss, Lsv, L1vv = Diffusivity.Lij(*Diffusivity.preene2betafree(kT, **thermaldef))
    #
    #     if self.logger.isEnabledFor(logging.DEBUG):
    #         self.logger.debug('Crystal: {}\n{}'.format(self.crystalname, Diffusivity))
    #         for Lname in ('Lvv', 'Lss', 'Lsv', 'L1vv'):
    #             self.logger.debug('{}:\n{}'.format(Lname, locals()[Lname]))
    #     # we leave out Lss since it is not, in fact, isotropic!
    #     for L in [Lvv, Lsv, L1vv]:
    #         self.assertTrue(np.allclose(L, L[0, 0] * np.eye(3), atol=1e-8),
    #                         msg='Diffusivity not isotropic?')
    #     # No solute drag, so Lsv = -Lvv; Lvv = normal vacancy diffusion
    #     # all correlation is in that geometric prefactor of Lss.
    #     self.assertTrue(np.allclose(Lvv, L0vv))
    #     self.assertTrue(np.allclose(-Lsv, L0vv))
    #     self.assertTrue(np.allclose(L1vv, 0.))
    #     correlmat = np.array([[self.correlx, 0, 0], [0, self.correlx, 0], [0, 0, self.correlz]])
    #     self.assertTrue(np.allclose(-Lss, np.dot(correlmat, Lsv), rtol=1e-7),
    #                     msg='Failure to match correlation ({}, {}), got {}, {}'.format(
    #                         self.correlx, self.correlz, -Lss[0, 0] / Lsv[0, 0], -Lss[2, 2] / Lsv[2, 2]))
    #     # test large_om2 version:
    #     self.assertEqualDiffusivity(Diffusivity, thermaldef, Diffusivity, thermaldef,
    #                                 diffuserargs2={'large_om2': 0}, msg='large omega test fail')

    def testHighOmega2(self):
        """Test that HCP with very high omega2 still produces symmetric diffusivity"""
        self.logger = logging.getLogger(__name__ + '.' +
                                        self.__class__.__name__ + '.' +
                                        inspect.currentframe().f_code.co_name)
        # Make a calculator with one neighbor shell
        kT = 1.
        Diffusivity = OnsagerCalc.VacancyMediatedMeta(self.crys, self.chem, self.sitelist, self.jumpnetwork, 2,
                                                      meta_sites = self.meta_sites, jumpnetwork2= self.jumpnetwork2)

        HCPtracer = {'preV': np.array([1.0, 1.0 / 10000000.0]), 'eneV': np.array([0.0, 0.0]),
                     'preT0': np.array([0.5, 1.0]),
                     'eneT0': np.array([0, 0]),
                     }

        HCPtracer.update(Diffusivity.maketracerpreene(**HCPtracer))
        HCPtracer['preT2'] = np.array([0.5, 0.5])

        #thermaldef = self.makeunitythermodict(Diffusivity)
        #thermaldef['preT2'] = 1e16*thermaldef['preT2']

        HCPtracer['preT2'] = 1e16 * HCPtracer['preT2']
        Lvv, Lss, Lsv, L1vv = Diffusivity.Lij(*Diffusivity.preene2betafree(kT, **HCPtracer))
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('Crystal: {}\n{}'.format(self.crystalname, Diffusivity))
            for Lname in ('Lvv', 'Lss', 'Lsv', 'L1vv'):
                self.logger.debug('{}:\n{}'.format(Lname, locals()[Lname]))
        for Lname in ('Lvv', 'Lss', 'Lsv', 'L1vv'):
            L = locals()[Lname]
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug('{}:\n{}'.format(Lname, L))
            for i in range(3):
                for j in range(i):
                    self.assertAlmostEqual(L[i,j], L[j,i],
                                           msg="{} not symmetric?\n{}".format(Lname, L))

