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
        
        if(np.isnan(tau_d[5]) or np.isnan(tau_d[0]) or np.isnan(tau_d[1])):
            #print("tau_d is nan")
            tau_d = np.zeros(6)
        if(tau_d[5]==0 and tau_d[0]==0 and tau_d[1]==0):
            #print("tau_d is zero")
            return np.zeros(n), np.zeros(n) 


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

        z0 = np.array([u_now[0], Fx1_now, Fy1_now, Fx2_now, Fy2_now])
        # print("z0:", z0)
        if(self.thrusters[0].u_max**2 - z0[0]**2 < 0):
            z0[0] = np.sign(z0[0]) * (self.thrusters[0].u_max - 1) 
        while(self.thrusters[1].u_max**2 - z0[1]**2 - z0[2]**2 < 0):
            z0[1], z0[2] = 0.99 * np.array([z0[1], z0[2]])
        while(self.thrusters[2].u_max**2 - z0[3]**2 - z0[4]**2 < 0):
            z0[3], z0[4] = 0.99 * np.array([z0[3], z0[4]])
        #building the cost function:
        def cost(z): #3x5 5x1 - 3x1
            s = B_e @ z - tau_d[[0,1,5]] #error
            return s.T @ Q @ s + z.T @ R @ z


        #actuator constraints:
        #TODO: Implement azimuth slewing constraints
        #TODO: Implement rate-aware allocation:
        constraints = [
            {'type': 'ineq', 'fun': lambda z: self.thrusters[0].u_max**2 - z[0]**2},  # tunnel thruster
            {'type': 'ineq', 'fun': lambda z: self.thrusters[1].u_max**2 - z[1]**2 - z[2]**2},  # azimuth thruster 1
            {'type': 'ineq', 'fun': lambda z: self.thrusters[2].u_max**2 - z[3]**2 - z[4]**2},  # azimuth thruster 2       
        ]

        result=minimize(cost, z0, method='SLSQP', constraints=constraints, options={
            'ftol': 1e-8,
            'maxiter': 2000,
            'disp': False
        })
        z = result.x

        if not result.success:
            #print(f"[allocator] t={t:.2f}s: SLSQP did not converge ({result.message}) — clipping to feasible set")
            pass

        if np.any(np.isnan(z)):
            # solver returned garbage; fall back to the (already-feasible) initial guess
            z = z0

        u_max_T = self.thrusters[0].u_max
        u_max_1 = self.thrusters[1].u_max
        u_max_2 = self.thrusters[2].u_max

        def clip_pair(fx, fy, umax):
            mag = np.hypot(fx, fy)
            if mag > umax and mag > 0:
                scale = umax / mag
                return fx * scale, fy * scale
            return fx, fy

        z[0] = np.clip(z[0], -u_max_T, u_max_T)
        z[1], z[2] = clip_pair(z[1], z[2], u_max_1)
        z[3], z[4] = clip_pair(z[3], z[4], u_max_2)

        u_T = z[0]
        FxA1, FyA1 = z[1], z[2]
        FxA2, FyA2 = z[3], z[4]
        """
        #recovering azimuth thrust magnitudes and angles from the solution:
        u_1=np.sqrt(FxA1**2+FyA1**2)
        u_2=np.sqrt(FxA2**2+FyA2**2)

        a_1=np.arctan2(FyA1,FxA1)
        a_2=np.arctan2(FyA2,FxA2)

        #make sure alpha is within [-pi, pi]:
        a_1=wrap_angle_pi(a_1)
        a_2=wrap_angle_pi(a_2)
        """
        def circ_dist(a, b):
            return abs(wrap_angle_pi(a - b))

        def signed_thrust(Fx, Fy, alpha_prev):
            u = np.hypot(Fx, Fy)
            a = wrap_angle_pi(np.arctan2(Fy, Fx))
            a_flip = wrap_angle_pi(a + np.pi)
            if circ_dist(a_flip, alpha_prev) < circ_dist(a, alpha_prev):
                return -u, a_flip
            return u, a

        u_1, a_1 = signed_thrust(FxA1, FyA1, alpha_now[1])
        u_2, a_2 = signed_thrust(FxA2, FyA2, alpha_now[2])

        u_cmd = np.array([u_T, u_1, u_2])
        alpha_cmd = np.array([np.pi / 2, a_1, a_2])

        u_now = u_cmd
        alpha_now = alpha_cmd
        #print("tau_d:", tau_d[[0,1,5]])
        tau_achieved = B_e @ np.array([u_T, FxA1, FyA1, FxA2, FyA2])
        #print("tau_achieved:", tau_achieved)

        return u_cmd, alpha_cmd
