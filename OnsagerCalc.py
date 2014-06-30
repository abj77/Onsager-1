"""
Onsager calculator module

Class to create an Onsager "calculator", which brings two functionalities:
1. determines *what* input is needed to compute the Onsager (mobility, or L) tensors
2. constructs the function that calculates those tensors, given the input values.

This class is designed to be combined with code that can, e.g., automatically
run some sort of atomistic-scale (DFT, classical potential) calculation of site
energies, and energy barriers, and then in concert with scripts to convert such data
into rates and probabilities, this will allow for efficient evaluation of transport
coefficients.

This implementation will be for vacancy-mediated solute diffusion on a Bravais lattice,
and assumes the dilute limit. The mathematics is based on a Green function solution
for the vacancy diffusion. The computation of the GF is outside the scope of this
particular module; however, clever uses of the GFcalc module can be attached to this
quite easily.
"""

# TODO: need to make sure we can read / write Onsager comp. information for optimal functionality (JSON?)

__author__ = 'Dallas R. Trinkle'

import numpy as np
import stars


class VacancyMediated:
    """
    A class to compute vacancy-mediated solute transport coefficients, specifically
    L_vv (vacancy diffusion), L_ss (solute), and L_sv (off-diagonal). As part of that,
    it determines *what* quantities are needed as inputs in order to perform this calculation.

    Currently for Bravais lattice only.
    """
    def __init__(self, jumpvect, groupops, Nthermo = 0):
        """
        Initialization; starts off with a set of jump vectors, group operations, and an optional
        range for thermodynamic interactions.

        Parameters
        ----------
        jumpvect : either list of array[3] or array[:, 3]
            list of jump vectors that are possible

        groupops : list of array[3, 3]
            point group operations

        Nthermo : integer, optional
            range of thermodynamic interactions, in terms of "shells", which is multiple
            summations of jumpvect
        """
        # make copies, as lists (there is likely a more efficient way to do this
        self.jumpvect = [v for v in jumpvect]
        self.groupops = [g for g in groupops]
        self.NNstar = stars.StarSet(self.jumpvect, self.groupops, 1)
        self.thermo = stars.StarSet(self.jumpvect, self.groupops)
        self.kinetic = stars.StarSet(self.jumpvect, self.groupops)
        self.GF = stars.StarSet(self.jumpvect, self.groupops)
        self.omega1 = stars.DoubleStarSet(self.kinetic)
        self.biasvec = stars.VectorStarSet(self.GF)
        self.Nthermo = 0
        self.generate(Nthermo)

    def generate(self, Nthermo):
        """
        Generate the necessary stars, double-stars, vector-stars, etc. based on the
        thermodynamic range.

        Parameters
        ----------
        Nthermo : integer
            range of thermodynamic interactions, in terms of "shells", which is multiple
            summations of jumpvect
        """
        if Nthermo == 0:
            self.Nthermo = 0
        if Nthermo == self.Nthermo:
            return
        self.thermo.generate(Nthermo)
        self.kinetic.combine(self.thermo, self.NNstar)
        self.omega1.generate(self.kinetic)
        # the following is a list of indices corresponding to the jump-type; so that if one
        # chooses *not* to calculate omega1, the corresponding omega0 value can be substituted
        # This is returned both for reference, and used for internal consumption
        self.omega1LIMB = [self.NNstar.starindex(self.kinetic.pts[p[0][0]] - self.kinetic.pts[p[0][1]])
                           for p in self.omega1.dstars]
        self.GF.combine(self.kinetic, self.kinetic)
        self.biasvec.generate(self.GF)
        # this is the list of points for the GF calculation; we need to add in the origin now:
        self.GFR = [np.zeros(3)] + [sR[0] for sR in self.GF.stars]

    def omega0list(self, Nthermo = None):
        """
        Return a list of endpoints for a vacancy jump, corresponding to omega0: no solute.
        Note: omega0list and omega2list are, by definition, the same. Defined by Stars.

        Parameters
        ----------
        Nthermo : integer, optional
            if set to some value, then we call generate(Nthermo) first.

        Returns
        -------
        omega0list : list of array[3]
            list of endpoints for a vacancy jump: we will expect rates for jumps from
            the origin to each of these endpoints as inputs for our calculation
        """
        if Nthermo != None:
            self.generate(Nthermo)
        return [s[0] for s in self.NNstar.stars]

    def interactlist(self, Nthermo = None):
        """
        Return a list of solute-vacancy configurations for interactions. The points correspond
        to a vector between a solute atom and a vacancy. Defined by Stars.

        Parameters
        ----------
        Nthermo : integer, optional
            if set to some value, then we call generate(Nthermo) first.

        Returns
        -------
        interactlist : list of array[3]
            list of vectors to connect a solute and a vacancy jump.
        """
        if Nthermo != None:
            self.generate(Nthermo)
        return [s[0] for s in self.thermo.stars]

    def GFlist(self, Nthermo = None):
        """
        Return a list of points for the vacancy GF calculation, corresponding to omega0: no solute.
        Defined by Stars.

        Parameters
        ----------
        Nthermo : integer, optional
            if set to some value, then we call generate(Nthermo) first.

        Returns
        -------
        GFlist : list of array[3]
            list of points to calculate the vacancy diffusion GF; this is based ultimately on omega0,
            and is the (pseudo)inverse of the vacancy hop rate matrix.
        """
        if Nthermo != None:
            self.generate(Nthermo)
        return self.GFR

    def omega1list(self, Nthermo = None):
        """
        Return a list of pairs of endpoints for a vacancy jump, corresponding to omega1:
        Solute at the origin, vacancy hopping between two sites. Defined by Double Stars.

        Parameters
        ----------
        Nthermo : integer, optional
            if set to some value, then we call generate(Nthermo) first.

        Returns
        -------
        omega1list : list of tuples of array[3]
            list of paired endpoints for a vacancy jump: the solute is at the origin,
            and these define start- and end-points for the vacancy jump.
        """
        if Nthermo != None:
            self.generate(Nthermo)
        return [(self.kinetic.pts[p[0][0]], self.kinetic.pts[p[0][1]]) for p in self.omega1.dstars], \
               self.omega1LIMB
