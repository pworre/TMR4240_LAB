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

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design.
"""
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from simulation.utils import wrap_angle_pi, wrap_angle_2pi
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
        '''part 2 rate aware allocation'''

        if(np.isnan(tau_d[5]) or np.isnan(tau_d[0]) or np.isnan(tau_d[1])):
            tau_d = np.zeros(6)
        if u_now is None or alpha_now is None:
            u_now = np.zeros(len(self.thrusters))
            alpha_now = np.array([np.pi/2, 0, 0]) 

        alpha_unwrapped = np.array([wrap_angle_2pi(alpha_now[0]), wrap_angle_2pi(alpha_now[1]), wrap_angle_2pi(alpha_now[2])])
        u_cmd, alpha_cmd, tau_achieved, success = allocate_part_2(tau_d/1000, u_now/1000, alpha_unwrapped, self, dt, t)
        if not success:
            return u_now, alpha_now
        #data_extraction(tau_achieved, tau_d, t, dt)
        return u_cmd, alpha_cmd

def data_extraction(tau_achieved, tau_d, t, dt):
    #print(dt)
    #print("time:", t)
    with open("allocator_results.csv", "a") as f:
        f.write(
            f"{tau_d[0]},{tau_d[1]},{tau_d[5]},"
            f"{tau_achieved[0]},{tau_achieved[1]},{tau_achieved[2]},\n"
        )
    return

def B_matrix(alpha_now, self):
    B_T=np.array([0,1,self.thrusters[0].x]).T
    B_A1=np.array([np.cos(alpha_now[1]),np.sin(alpha_now[1]), self.thrusters[1].x*np.sin(alpha_now[1])-self.thrusters[1].y*np.cos(alpha_now[1])]).T
    B_A2=np.array([np.cos(alpha_now[2]),np.sin(alpha_now[2]), self.thrusters[2].x*np.sin(alpha_now[2])-self.thrusters[2].y*np.cos(alpha_now[2])]).T
    B_e = np.column_stack([B_T, B_A1, B_A2])

    if(np.linalg.matrix_rank(B_e)<3):
        print("B is not full rank, check the actuator configuration matrix")
        return None

    singular_values = np.linalg.svd(B_e, compute_uv=False)
    condition_number = singular_values[0] / singular_values[-1]
    #print(f"Condition number: {condition_number:.2e}")
    return B_e

def allocate_part_2(tau_d, u_now, alpha_now, self,dt, t): #part 2 allocation rate aware
    '''
    u_now and tau_d in kNs, working with kNs, output in Ns

    alpha_now: unwrapped angles in [0, 2pi], gets wrapped after the optimization
    '''
    B_e = B_matrix(alpha_now, self)
    if B_e is None:
        return u_now, alpha_now, B_e @ u_now, False #no solution found, keep previous state
    
    Q=np.diag([1100*1e6,1100*1e6,300*1e6]) #error penalty
    P=np.diag([0.1,0.1,0.1]) #thruster magnitude penalty
    R = np.diag([0.1,0.1]) #azimuth angle penalty
    
    def cost(x): #x = [delta_z, delta_alpha, s]
        delta_z = x[0:3]
        delta_alpha = x[3:5]
        s = x[5:8]
        z = z = u_now + delta_z
        cost=s.T @ Q @ s + z.T @ P @ z + delta_alpha.T @ R @ delta_alpha
        return cost/1e6

    def thrust_constraint(x): #-u_max <= z <= u_max
        delta_z = x[0:3]
        z = u_now + delta_z
        u_max = np.array([
            thruster.u_max / 1000
            for thruster in self.thrusters[:3]
        ])
        #(u_max - z) >= 0 and (z - (-u_max)) >= 0
        lower_bound = z - (-u_max)
        upper_bound = u_max - z
        return np.concatenate([lower_bound, upper_bound])

    def wrench_constraint(x):
        delta_z = x[0:3]
        delta_alpha = x[3:5]
        s = x[5:8]
        z = u_now + delta_z
        alpha = np.array([alpha_now[0],alpha_now[1] + delta_alpha[0],alpha_now[2] + delta_alpha[1]])

        return B_matrix(alpha, self) @ z - tau_d[[0, 1, 5]] - s

    def singularity_constraint(x):
        delta_alpha = x[3:5]

        a1 = alpha_now[1] + delta_alpha[0]
        a2 = alpha_now[2] + delta_alpha[1]

        eps1 = 0.25*dt*self.thrusters[1].rot_speed
        eps2 = 0.25*dt*self.thrusters[1].rot_speed

        c=(np.cos(a1)**2 / np.sin(eps1)**2+ np.cos(a2)**2 / np.sin(eps2)**2- 1)

        return c
        

        

    constraints = [{"type": "ineq","fun": thrust_constraint},{"type": "eq","fun": wrench_constraint},{"type": "ineq","fun": singularity_constraint}]

    bounds = []
    for thruster in self.thrusters[:3]: #thrust rate
        limit = thruster.u_rate * dt / 1000
        bounds.append((-limit, limit))

    for thruster in self.thrusters[1:3]: #angle rate
        limit = thruster.rot_speed * dt
        bounds.append((-limit, limit))

    for _ in range(3): #slack variable
        bounds.append((None, None))


    s_initial = B_matrix(alpha_now, self) @ u_now - tau_d[[0, 1, 5]] #start from previous s
    x0 = np.concatenate([np.zeros(3), np.zeros(2), s_initial])

    result=minimize(cost, x0, method='SLSQP', constraints=constraints, bounds=bounds, options={
            'ftol': 1e-8,
            'maxiter': 500,
            'disp': False
        })

    res=result.x # x:[delta_z, delta_alpha, s]

    u_cmd, alpha_cmd= clip(alpha_now, u_now*1000, res[3:5], res[0:3]*1000, self, dt)
    tau_achieved = B_matrix(alpha_cmd, self) @ u_cmd
    for i in range(3):
        alpha_cmd[i] = wrap_angle_pi(alpha_cmd[i])
    
    #print(f"u_cmd: {u_cmd}, alpha_cmd: {alpha_cmd}, tau_achieved: {tau_achieved}, tau_d: {tau_d[[0, 1, 5]]}, t: {t}")
    return u_cmd, alpha_cmd, tau_achieved, True 


def feasible(z, limit, self):
    '''
    input in Ns for part 2
    input in kNs for part 1
    returns True if the solution is feasible, no azimuth transformation part 2
    '''
    if len(z) ==3: 
        if np.abs(z[0])-limit < self.thrusters[0].u_max and np.abs(z[1])-limit<self.thrusters[1].u_max and np.abs(z[2])-limit<self.thrusters[2].u_max:
            return True
        else:
            return False
    #check if solution matches thruster limits part 1 method
    magA1=np.hypot(z[1], z[2])-limit
    magA2=np.hypot(z[3], z[4])-limit
    if np.abs(z[0])-limit*0.5>self.thrusters[0].u_max/1000 or magA1>self.thrusters[1].u_max/1000 or magA2>self.thrusters[2].u_max/1000:
        return False
    return True


def clip(alpha_now, u_now, delta_alpha, delta_u, self, dt):
    '''
    make sure solution is within the rate limits and absolute limits
    
    input and output angles are unwrapped
    input in Ns output in Ns
    '''

    u_rate_limits = np.array([thruster.u_rate * dt for thruster in self.thrusters[:3]])
    alpha_rate_limits = np.array([thruster.rot_speed * dt for thruster in self.thrusters[1:3]])
    u_limits = np.array([thruster.u_max for thruster in self.thrusters[:3]])
    

    delta_u = np.clip(delta_u, -u_rate_limits, u_rate_limits)
    delta_alpha = np.clip(delta_alpha, -alpha_rate_limits, alpha_rate_limits)

    u_cmd = np.clip(u_now + delta_u, -u_limits, u_limits)
    alpha_cmd = np.array([alpha_now[0], alpha_now[1] + delta_alpha[0], alpha_now[2] + delta_alpha[1]])

    return u_cmd, alpha_cmd