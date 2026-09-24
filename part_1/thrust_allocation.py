"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.
"""
from typing import List, Optional, Tuple
from unittest import result
import numpy as np
from scipy.optimize import minimize
from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi

class ThrustAllocator:
    """Template for student thrust allocation.
    """

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)
        '''To better condition the problem I am working with kNs'''
     
        if(np.isnan(tau_d[5]) or np.isnan(tau_d[0]) or np.isnan(tau_d[1])):
            print("tau_d is nan")
            tau_d = np.zeros(6)

        #tunnel thruster (constant):
        B_T=np.array([0,1,self.thrusters[0].x]).T
        #azimuth thruster 1 (linearization):
        B_FxA1=np.array([1,0, -self.thrusters[1].y]).T
        B_FyA1=np.array([0,1, self.thrusters[1].x]).T
        #azimuth thruster 2 (linearization):
        B_FxA2=np.array([1,0, -self.thrusters[2].y]).T
        B_FyA2=np.array([0,1, self.thrusters[2].x]).T
        #creating the actuator configuration matrix
        B_e = np.column_stack([B_T, B_FxA1, B_FyA1, B_FxA2, B_FyA2])

        #check rank(B)=3:
        if(np.linalg.matrix_rank(B_e)<3):
            print("B is not full rank, check the actuator configuration matrix")
            return np.zeros(n), np.zeros(n)

        #weights for the cost function:
        Q=np.diag([100,100,100]) #error penalty
        R = np.diag([
            1.0 / self.thrusters[0].u_max**2,
            1.0 / self.thrusters[1].u_max**2,
            1.0 / self.thrusters[1].u_max**2,
            1.0 / self.thrusters[2].u_max**2,
            1.0 / self.thrusters[2].u_max**2,
        ])

        if u_now is None and alpha_now is None:
            u_now = np.zeros(n)
            alpha_now = np.zeros(n)

        Fx1_now = u_now[1] * np.cos(alpha_now[1])
        Fy1_now = u_now[1] * np.sin(alpha_now[1])
        Fx2_now = u_now[2] * np.cos(alpha_now[2])
        Fy2_now = u_now[2] * np.sin(alpha_now[2])

        z0 = np.array([u_now[0], Fx1_now, Fy1_now, Fx2_now, Fy2_now])/1000

        #building the cost function:
        def cost(z): #3x5 5x1 - 3x1
            s = B_e @ z - tau_d[[0,1,5]]/1000 #error
            return s.T @ Q @ s + z.T @ R @ z

        constraints = [
            {'type': 'ineq',
            'fun': lambda z: self.thrusters[0].u_max/1000 - np.abs(z[0])},
            {'type': 'ineq',
            'fun': lambda z: self.thrusters[1].u_max/1000 - np.hypot(z[1], z[2])},
            {'type': 'ineq',
            'fun': lambda z: self.thrusters[2].u_max/1000 - np.hypot(z[3], z[4])},
        ]

        result=minimize(cost, z0, method='SLSQP', constraints=constraints, options={
            'ftol': 1e-8,
            'maxiter': 2000,
            'disp': False
        })
        results=z=result.x

        if feasible(results, 1e-6, self):
            #print("Thrust matches controller demand")
            z=results
            u_cmd, alpha_cmd = recover_thrust(z)
            return u_cmd, alpha_cmd
        else:
            z=saturation_mode(B_e, tau_d, self)

        u_cmd, alpha_cmd = recover_thrust(z)
        return u_cmd, alpha_cmd


def saturation_mode(B_e, tau_d, self):
    tau = tau_d[[0, 1, 5]]/1000

    def cost(x):
        return -x[5]       

    def equality(x):
        z = x[:5]
        k = x[5]
        return B_e @ z - k * tau

    constraints = [
        # B_e z = lambda * tau_d
        {'type': 'eq', 'fun': equality},
        {'type': 'ineq','fun': lambda x: (self.thrusters[0].u_max/1000)**2 - x[0]**2},
        {'type': 'ineq','fun': lambda x: (self.thrusters[1].u_max/1000)**2 - x[1]**2 - x[2]**2},
        {'type': 'ineq','fun': lambda x: (self.thrusters[2].u_max/1000)**2 - x[3]**2 - x[4]**2},

        # 0 <= k scalar <= 1
        {'type': 'ineq', 'fun': lambda x: x[5]},
        {'type': 'ineq', 'fun': lambda x: 1.0 - x[5]},
    ]
    #pseudo inverse without constrains
    z_unit = np.linalg.pinv(B_e) @ tau
    scales = [
        self.thrusters[0].u_max/1000 / abs(z_unit[0]),
        self.thrusters[1].u_max/1000 / np.hypot(z_unit[1], z_unit[2]),
        self.thrusters[2].u_max/1000 / np.hypot(z_unit[3], z_unit[4])
    ]
    #creating a valid initial scaler to start from
    k0 = min(1.0, *scales) * 0.5

    z0 = k0 * z_unit
    x0 = np.r_[z0, k0]
    result = minimize(cost, x0 , method='SLSQP', constraints=constraints, options={'ftol': 1e-8,'maxiter': 2000,'disp': False})

    z = result.x[:5]
    k = result.x[5]

    if feasible(z,1e-3, self):
        print("Warning: Saturated thrust!") 
        return z
    
    print("Warning: Zero Thrust!") #no solution found
    return np.zeros(5)

def feasible(z, limit, self):
    #check if solution matches thruster limits
    magA1=np.hypot(z[1], z[2])-limit
    magA2=np.hypot(z[3], z[4])-limit
    if z[0]-limit*0.5>self.thrusters[0].u_max/1000 or magA1>self.thrusters[1].u_max/1000 or magA2>self.thrusters[2].u_max/1000:
        return False
    return True

def recover_thrust(z):
    z=1000*z #back to N
    u_T = z[0]
    FxA1, FyA1 = z[1], z[2]
    FxA2, FyA2 = z[3], z[4]
            
    #recovering azimuth thrust magnitudes and angles from the solution:
    u_1=np.hypot(FxA1,FyA1)
    u_2=np.hypot(FxA2,FyA2)
    
    a_1=np.arctan2(FyA1,FxA1)
    a_2=np.arctan2(FyA2,FxA2)
    
    #make sure alpha is within [-pi, pi]:
    a_1=wrap_angle_pi(a_1)
    a_2=wrap_angle_pi(a_2)
    u_cmd = np.array([u_T, u_1, u_2])
    alpha_cmd = np.array([np.pi / 2, a_1, a_2])
    return u_cmd, alpha_cmd
