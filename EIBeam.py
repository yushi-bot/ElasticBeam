"""
ElasticBeam class written by Yerr (2025/05/15)
A large deformation Cantilever beam class with loading at free side and middle
Newmark-beta method was adopted to integrate deflection and rotation angle
"""

import numpy as np
import matplotlib.pyplot as plt
import math

#---------CONSTANT--------------
LARGE_DEFORMATION = "LARGE_DEFORMATION"
SMALL_DEFORMATION = "SMALL_DEFORMATION"

class EIBeam:

#----------geometric and material parameters------------
    L:float # Beam length (m)
    E:float # Young's modulus (Pa)
    I:float # Second moment of area (m^4)

#---------------------------------Load-------------------
    P:float # End load
    F:float # mid load
    L1:float # mid load position

#-----------------------------Discrement-------------------
    dx:float    # distinct length
    x = []      # coordinates
    M = []      # moments
    w = []      # deflection
    dw = []     # # 1-order deflection
    theta = []  # rotation angel
    ddw = []    # 2-order deflection

    def __init__(self, length:float, modulus:float, inertia:float):
        """
        :param length: Beam length (m)
        :param modulus: Young's modulus (Pa)
        :param inertia: Second moment of area (m^4)
        """
        self.L = length
        self.E = modulus
        self.I = inertia

    def copyBeamWithDiffLoad(self, endLoad:float, midLoad:float):
        """
        copy the beam object, but set its load to new load
        :param endLoad: new end load
        :param midLoad: new middle load
        :return: new beam
        """
        newBeam = EIBeam(self.L,self.E,self.I)
        newBeam.L1 = self.L1
        newBeam.P = endLoad
        newBeam.F = midLoad
        newBeam.discrete(self.dx)
        return newBeam



    def setLoadPosition(self, position:float):
        """
        Set middle load position
        :param position: coordinates of load (from fixed side)
        :return:
        """
        self.L1 = position

    def setLoad(self, endLoad:float, midLoad:float):
        """
        set load
        :param endLoad: loading force at free end
        :param midLoad: loading force at middle load position
        :return:
        """
        self.P = endLoad
        self.F = midLoad

    def discrete(self, dis_length:float):
        """
        discretize the beam
        :param dis_length: discretize length
        :return:
        """
        self.dx = dis_length
        N = int(self.L / self.dx) + 1
        self.x = np.linspace(0, self.L, N)
        self.M = np.zeros(N)  # Moment
        self.w = np.zeros(N)  # Deflection
        self.dw = np.zeros(N)  # 1-order deflection
        self.ddw = np.zeros(N)  # 2-order deflection
        self.calcMoment()

    def calcMoment(self):
        """
        calculate discretize moment
        """
        n = len(self.x)
        for i in range(n - 1):
            M1 = self.P * (self.L - self.x[i])
            M2 = self.F * (self.L1 - self.x[i])
            self.M[i] = M1
            if self.x[i] < self.L1:
                self.M[i] += M2

    def calcDDW(self,M, dw, formulation=LARGE_DEFORMATION):
        """
        calculate 2 order derivative of w ddw with Large deformation equations
        :param M: moment
        :param dw: 1 order derivative of w
        :param formulation: formulation, LARGE_DEFORMATION or SMALL_DEFORMATION
        :return: ddw: 2 order derivative of w
        """
        if SMALL_DEFORMATION == formulation:
            return self.calcDDwSmall(M,dw)
        else:
            return self.calcDDwLarge(M,dw)

    def calcDDwLarge(self, M, dw):
        """
        calculate 2 order derivative of w ddw with Large deformation equations
        :param M: moment
        :param dw: 1 order derivative of w
        :return: ddw: 2 order derivative of w
        """
        return M / self.E / self.I *math.pow(1 + dw * dw, 1.5)

    def calcDDwSmall(self, M, dw): # small deformation equations
        """
        calculate 2 order derivative of w ddw with small deformation equations
        :param M: moment
        :param dw: 1 order derivative of w
        :return: ddw: 2 order derivative of w
        """
        return M / self.E / self.I

    def calcDeflectionWithM(self, formulation=LARGE_DEFORMATION):
        n = len(self.x)
        for i in range(n - 1):  # Newmark-beta method, gamma = 0.5, beta = 0.25
            dx = self.x[i + 1] - self.x[i]
            dw_i = self.dw[i]
            ddw_i0 = self.calcDDW(self.M[i], dw_i, formulation)

            dw_i1 = dw_i + ddw_i0 * dx
            ddw_i1 = self.calcDDW(self.M[i + 1], dw_i1, formulation)
            dw_i2 = dw_i + ddw_i1 * dx

            self.ddw[i] = ddw_i0
            self.dw[i + 1] = (dw_i1 + dw_i2) / 2
            self.w[i + 1] = self.w[i] + self.dw[i] * dx + (ddw_i0 + ddw_i1) * 0.25 * dx * dx
        self.theta = np.asin(self.dw)


    def calcDeflection(self, formulation=LARGE_DEFORMATION):
        self.calcDeflectionWithM(formulation)
        M = np.copy(self.M)
        tol = 1e-2
        M_max = np.max(M)
        M_min = np.min(M)
        M2 = self.modifyMWithTheta()
        if  math.fabs(M_max) > math.fabs(M_min):
            M2_max = np.max(M2)
            err = M2_max - M_max
            errTol = tol*math.fabs(M_max)
        else:
            M2_min = np.min(M2)
            err = M2_min - M_min
            errTol = tol * math.fabs(M_min)

        while not( -errTol < err < errTol):
            #break
            self.calcDeflectionWithM(formulation)
            M = np.copy(self.M)
            tol = 1e-2
            M_max = np.max(M)
            M_min = np.min(M)
            M2 = self.modifyMWithTheta()
            if math.fabs(M_max) > math.fabs(M_min):
                M2_max = np.max(M2)
                err = M2_max - M_max
                errTol = tol * math.fabs(M_max)
            else:
                M2_min = np.min(M2)
                err = M2_min - M_min
                errTol = tol * math.fabs(M_min)




    def modifyMWithTheta(self):
        n = len(self.x)
        self.M[-1] = 0
        for i in range(1,n):
            t = n-i
            dx = self.x[t] - self.x[t-1]
            dx *= math.cos(self.theta[t-1])
            self.M[t-1] = self.M[t] + self.P * dx
            if self.x[t-1] < self.L1:
                self.M[t-1] += self.F * dx

        return self.M


    def plot(self):
        """
        draw plot
        :return:
        """
        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.plot(self.x, self.M, 'b', linewidth=2)
        plt.xlabel('x (m)')
        plt.ylabel('M (N·m)')
        plt.title('Moment')
        plt.grid(True)

        plt.figure(1)
        plt.subplot(2, 2, 2)
        plt.plot(self.x, self.ddw, 'r', linewidth=2)
        plt.xlabel('x (m)')
        plt.ylabel('w\'\' ')
        plt.title('two order derivative')
        plt.grid(True)

        plt.figure(1)
        plt.subplot(2, 2, 3)
        plt.plot(self.x, self.w, 'r', linewidth=2)
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.title('Deflection')
        plt.grid(True)

        plt.figure(1)
        plt.subplot(2, 2, 4)
        plt.plot(self.x, self.theta, 'r', linewidth=2)
        plt.xlabel('x (m)')
        plt.ylabel('theta (rad)')
        plt.title('rotation angle')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Elastica (Euler) beam parameters
    L = 1.0  # Beam length (m)
    E = 210e9  # Young's modulus (Pa)
    I = 1e-6  # Second moment of area (m^4)
    P = 400000.0  # End compressive load (N)
    F = -1200000  # mid load
    L1 = 0.5  # mid position

    L = 0.2  # 40 cm
    E = 210e9  # steel
    b = 0.03  # 3cm
    h = 0.001  # 0.8mm
    I = 1 / 12 * b * h * h * h
    P = 0  # initial load
    L1 = 0.15  # position
    F = -30  # initial load

    dx = L * 0.0001  # Step size along length

    beam = EIBeam(length=L, modulus=E, inertia=I)
    beam.setLoadPosition(L1)
    beam.setLoad(endLoad=P,midLoad=F)
    beam.discrete(dx)
    beam.calcDeflection(formulation=LARGE_DEFORMATION)
    beam.plot()


"""
# Elastica (Euler) beam parameters
L = 1.0              # Beam length (m)
E = 210e9            # Young's modulus (Pa)
I = 1e-6             # Second moment of area (m^4)
P = 400000.0           # End compressive load (N)
F = -1200000              # mid load
L1 = 0.5             # mid position

# Discretization
dx = L * 0.0001                     # Step size along length
N = int(L / dx) + 1                 # Number of points
x = np.linspace(0, L, N)
M = np.zeros(N)                     # Moment
w = np.zeros(N)                     # Deflection
dw = np.zeros(N)                    # 1-order deflection
ddw = np.zeros(N)                   # 2-order deflection

w2 = np.zeros(N)
w3 = np.zeros(N)
dw2 = np.zeros(N)
dw3 = np.zeros(N)
ddw2 = np.zeros(N)
ddw3 = np.zeros(N)

# Forward Euler Method
for i in range(N-1):
    M1 = P * (L - x[i])
    M2 = F * (L1 - x[i])
    M[i] = M1
    if x[i] < L1:
        M[i] += M2

    # Large deformation equations
    ddw[i] = M[i]/E/I * math.pow( (1+dw[i])*(1+dw[i]), 1.5)
    dw[i+1] = dw[i] + ddw[i] * dx

    ddw[i] += M[i] / E / I * math.pow((1 + dw[i+1]) * (1 + dw[i+1]), 1.5)
    ddw[i]  /= 2
    dw[i+1] = dw[i] + ddw[i] * dx
    w[i+1] = w[i] + 0.5 * dx * (dw[i] + dw[i + 1])

    # small deformation equations
    ddw2[i] = M[i] / E / I
    dw2[i + 1] = dw2[i] + ddw2[i] * dx
    w2[i + 1] = w2[i] + 0.5 * dx * (dw2[i] + dw2[i+1])

    ddw3[i + 1] = M[i]/E/I * math.pow( (1+dw3[i])*(1+dw3[i]), 1.5)
    dw3[i + 1] = dw3[i] + ddw3[i] * dx
    w3[i + 1] = w3[i] + dx * dw3[i]

# calculate theta
theta = np.asin(dw)
theta2 = np.asin(dw2)
theta3 = np.asin(dw3)


plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(x, M, 'b', linewidth=2)
plt.xlabel('x (m)')
plt.ylabel('M (N·m)')
plt.title('Moment')
plt.grid(True)

plt.figure(1)
plt.subplot(2, 2, 2)
plt.plot(x, ddw, 'r', linewidth=2, label="large def")
plt.plot(x, ddw2,'b', linewidth=2, label="small def")
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('w\'\' ')
plt.title('two order derivative')
plt.grid(True)
#plt.axis('equal')

plt.figure(1)
plt.subplot(2, 2, 3)
plt.plot(x, w, 'r', linewidth=2, label="large def")
plt.plot(x, w2, 'b', linewidth=2, label="small def")
plt.plot(x, w3, 'g', linewidth=2, label="1 order")
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Deflection')
plt.grid(True)


plt.figure(1)
plt.subplot(2, 2, 4)
plt.plot(x, theta,'r', linewidth=2, label="large def")
plt.plot(x, theta2,'b', linewidth=2, label="small def")
plt.plot(x, theta3,'g', linewidth=2, label="1 order")
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('theta (rad)')
plt.title('rotation angle')
plt.grid(True)
plt.show()
"""