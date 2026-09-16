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

class ThrustAllocator:
    """Template for student thrust allocation."""

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

        # TODO: Replace this placeholder with your thrust allocation algorithm.
        # The placeholder commands zero thrust and alpha for all thrusters.

        
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
        Q=np.identity(3) #error penalty
        R = np.diag([
            12.0 / self.thrusters[0].u_max**2,
            1.0 / self.thrusters[1].u_max**2,
            1.0 / self.thrusters[1].u_max**2,
            1.0 / self.thrusters[2].u_max**2,
            1.0 / self.thrusters[2].u_max**2,
        ])

        if u_now is None and alpha_now is None:
            u_now[0] = 0
            Fx1_now = 0
            Fy1_now = 0
            Fx2_now = 0
            Fy2_now = 0
        else:
            Fx1_now = u_now[1] * np.cos(alpha_now[1])
            Fy1_now = u_now[1] * np.sin(alpha_now[1])
            
            Fx2_now = u_now[2] * np.cos(alpha_now[2])
            Fy2_now = u_now[2] * np.sin(alpha_now[2])

        z0 = np.array([u_now[0]-10, Fx1_now, Fy1_now, Fx2_now, Fy2_now])
        #print("z0:", z0)

        #building the cost function:
        def cost(z): #3x5 5x1 - 3x1
            s = B_e @ z - tau_d[[0,1,5]] #error
            return s.T @ Q @ s + z.T @ R @ z

        #actuator constraints:
        #TODO: Implement azimuth slewing constraints
        #TODO: Implement rate-aware allocation:
        constraints = [
            {'type': 'ineq', 'fun': lambda z: self.thrusters[0].u_max - z[0]},  # tunnel thruster
            {'type': 'ineq', 'fun': lambda z: self.thrusters[0].u_max + z[0]},
            {'type': 'ineq', 'fun': lambda z: self.thrusters[1].u_max**2 - z[1]**2 - z[2]**2},  # azimuth thruster 1
            {'type': 'ineq', 'fun': lambda z: self.thrusters[2].u_max**2 - z[3]**2 - z[4]**2}  # azimuth thruster 2       
        ]

        #solve:
        if u_now is None and alpha_now is None:
            u_now = np.zeros(n)
            alpha_now = np.zeros(n)
        result=minimize(cost, z0, method='SLSQP', constraints=constraints, options={
            'ftol': 1e-8,
            'maxiter': 2000,
            'disp': True
        })
        '''if not result.success:
            raise RuntimeError(
                f"Thruster allocation failed: {result.message}"
            )'''
        if not result.success:
            print(f"Allocator warning: {result.message}")
        #print("message:", result.message)
        #print("z:", result.x)
        #print("cost:", result.fun)

        

        z = result.x
        u_T = z[0]

        FxA1, FyA1 = z[1], z[2]
        FxA2, FyA2 = z[3], z[4]

        #recovering azimuth thrust magnitudes and angles from the solution:
        u_1=np.sqrt(FxA1**2+FyA1**2)
        u_2=np.sqrt(FxA2**2+FyA2**2)

        a_1=np.arctan2(FyA1,FxA1)
        a_2=np.arctan2(FyA2,FxA2)

        #make sure alpha is within [-pi, pi]:
        a_1=(np.pi+a_1)%(2*np.pi)-np.pi
        a_2=(np.pi+a_2)%(2*np.pi)-np.pi
        


        u_cmd = [u_T, u_1, u_2]
        alpha_cmd = [np.pi / 2, a_1, a_2]

        u_now = u_cmd
        alpha_now = alpha_cmd

        return u_cmd, alpha_cmd
