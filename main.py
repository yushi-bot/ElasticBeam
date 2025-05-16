"""
Main program written by Yerr (2025/05/15)
Newton-Lawson method was adopted to calculate loading force.
"""

from EIBeam import *
import numpy as np
import math




def calcBC1Error(beam:EIBeam, r:float):
    """
    Boundary condition 1, w|x=L = r
    :param beam: EIBeam object
    :param r: fixed deflection of beam at x=L
    :return: difference of r and calculated deflection
    """
    return beam.w[-1] - r

def calcBC2Error(beam:EIBeam, phi:float):
    """
    Boundary condition 2, theta|x=L = phi
    :param beam: EIBeam object
    :param phi: fixed rotation angel of beam at x=L
    :return: difference of sin(phi) and calculated dw
    """
    return beam.dw[-1] - math.sin(phi)

def calcDerivativeMatrix(beam:EIBeam, BC1, BC2, r, phi):
    """
    calculate derivative of BC errors
    :param beam: beam object
    :param BC1: error of BC1
    :param BC2: error of BC2
    :param r: Boundary condition 1, w|x=L = r
    :param phi: Boundary condition 2, theta|x=L = phi
    :return:
    """
    dp = 1e-3
    df = 1e-3
    beam_dp = beam.copyBeamWithDiffLoad(beam.P + dp, beam.F)
    beam_dp.calcDeflection()
    dBC1_dp = (calcBC1Error(beam_dp, r) - BC1) / dp
    dBC2_dp = (calcBC2Error(beam_dp, phi) - BC2) / dp

    beam_df = beam.copyBeamWithDiffLoad(beam.P, beam.F + df)
    beam_df.calcDeflection()
    dBC1_df = (calcBC1Error(beam_df, r) - BC1) / df
    dBC2_df = (calcBC2Error(beam_df, phi) - BC2) / df

    return np.array([[dBC1_dp,dBC1_df],
                     [dBC2_dp,dBC2_df]])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    L = 0.3  # 30 cm
    E = 210e9 # steel
    b = 0.03 # 3cm
    h = 0.002 # 2mm
    I = 0.5 * b * h * h * h
    P0 = -10      # initial load
    L1 = 0.25   # position
    F0 = 10      # initial load

    r = 0.03 # 30mm
    phi = math.atan(0.285)
    tol = 1e-2  # 1%
    """
    L = 0.4  # 40 cm
    E = 210e9  # steel
    b = 0.03  # 3cm
    h = 0.0008  # 0.8mm
    I = 1/12 * b * h * h * h
    P0 = -10  # initial load
    L1 = 0.35  # position
    F0 = 10  # initial load

    r = 0.025  # 25mm
    phi = math.atan(0.285)
    tol = 1e-2  # 1%

    BC1tol = r * tol
    BC2tol = math.fabs(phi) * tol


    dx = L * 0.0001  # Step size along length

    # initialize the beam object
    beam = EIBeam(length=L, modulus=E, inertia=I)
    beam.setLoadPosition(L1)
    beam.setLoad(endLoad=P0, midLoad=F0)
    beam.discrete(dx)
    beam.calcDeflection(formulation=LARGE_DEFORMATION)

    BC1error = calcBC1Error(beam, r)
    BC2error = calcBC2Error(beam, phi)
    P = P0
    F = F0

    # calculate P and F with Newton-Lawson method
    while not( -BC1tol<BC1error<BC1tol and -BC2tol<BC2error<BC2tol):

        # calculate derivative
        derivativeMatrix = calcDerivativeMatrix(beam, BC1error, BC2error, r, phi)
        errorMatrix = np.array([[BC1error],
                               [BC2error]])
        derivative_inv = np.linalg.inv(derivativeMatrix)
        dForceMatrix = np.linalg.matmul( derivative_inv,errorMatrix)

        dp = dForceMatrix[0][0]
        df = dForceMatrix[1][0]

        P = P-dp
        F = F-df
        beam = beam.copyBeamWithDiffLoad(P,F)
        beam.calcDeflection()
        BC1error = calcBC1Error(beam, r)
        BC2error = calcBC2Error(beam, phi)

    print("P = %.4f" % P, "F = %.4f" % F)
    beam.plot()




