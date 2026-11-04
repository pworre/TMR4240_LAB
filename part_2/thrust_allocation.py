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
from simulation.utils import wrap_angle_pi
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
        if(np.isnan(tau_d[5]) or np.isnan(tau_d[0]) or np.isnan(tau_d[1])):
            print("tau_d is nan")
            tau_d = np.zeros(6)
        if u_now is None or alpha_now is None:
            u_now = np.zeros(len(self.thrusters))
            alpha_now = np.zeros(len(self.thrusters))
        #TODO angle wrapping for alpha_now and alpha_cmd
        #u_cmd, alpha_cmd, B_e, tau_achieved=allocate_part_1(tau_d, u_now, alpha_now, self)
        u_cmd, alpha_cmd, tau_achieved, success = allocate_part_2(tau_d, u_now, alpha_now, self, dt, t)
        model=2
        if not success:
            #print("Fallback to part 1 allocation")
            u_cmd, alpha_cmd, B_e, tau_achieved, model = allocate_part_1(tau_d, u_now, alpha_now, self)
        data_extraction(tau_achieved, tau_d, model, t)
        ok=dynamics(u_now, alpha_now, u_cmd, alpha_cmd,dt ,self ,t, model)


        #TODO clip numerical solutions to exact limits
        return u_cmd, alpha_cmd

def data_extraction(tau_achieved, tau_d, model, t):
    #print("time:", t)
    with open("allocator_results.csv", "a") as f:
        f.write(
            f"{model},"
            f"{tau_d[0]},{tau_d[1]},{tau_d[5]},"
            f"{tau_achieved[0]},{tau_achieved[1]},{tau_achieved[2]},\n"
        )
    return

def B_matrix(alpha_now, self):
    #B_e no lineraization of the azimuth angles
    B_T=np.array([0,1,self.thrusters[0].x]).T
    B_A1=np.array([np.cos(alpha_now[1]),np.sin(alpha_now[1]), self.thrusters[1].x*np.sin(alpha_now[1])-self.thrusters[1].y*np.cos(alpha_now[1])]).T
    B_A2=np.array([np.cos(alpha_now[2]),np.sin(alpha_now[2]), self.thrusters[2].x*np.sin(alpha_now[2])-self.thrusters[2].y*np.cos(alpha_now[2])]).T
    B_e = np.column_stack([B_T, B_A1, B_A2])
    #check rank(B)=3:
    if(np.linalg.matrix_rank(B_e)<3):
        print("B is not full rank, check the actuator configuration matrix")
        return np.zeros(self.n), np.zeros(self.n)
    return B_e

def allocate_part_2(tau_d, u_now, alpha_now, self,dt, t): #part 2 allocation rate aware
    '''Thrust input in Ns ->working with kNs -> output in Ns'''

    u_now_KN = u_now/1000
    B_e = B_matrix(alpha_now, self)
    if(np.linalg.matrix_rank(B_e)<3):
        print("B is not full rank, check the actuator configuration matrix")
        return np.zeros(self.n), np.zeros(self.n), np.zeros(3)

    #weights for the cost function:
    Q=np.diag([100*1e6,100*1e6,100*1e6]) #error penalty
    #Q=np.diag([100,100,100])
    P=np.diag([1,1,1]) #thruster magnitude penalty
    R = np.diag([1,1]) #azimuth angle penalty
    
    #cost function, thrust, slack variable, angle:
    def cost(x): #x = [delta_z, delta_alpha, s]
        delta_z = x[0:3]
        delta_alpha = x[3:5]
        s = x[5:8]
        z = u_now_KN + delta_z
        return s.T @ Q @ s + z.T @ P @ z + delta_alpha.T @ R @ delta_alpha

    def thrust_constraint(x): #thruster absolute magnitude limit
        delta_z = x[0:3]
        z = u_now_KN + delta_z
        u_max = np.array([
            thruster.u_max / 1000
            for thruster in self.thrusters[:3]
        ])
        return u_max - np.abs(z)

    def azimuth_rate_constraint(x): #azimuth rotation rate limit
        delta_alpha = x[3:5]
        alpha_max = np.array([(self.thrusters[1].rot_speed * dt - abs(delta_alpha[0])), (self.thrusters[2].rot_speed * dt - abs(delta_alpha[1]) )])
        return alpha_max

    def thrust_rate_constraint(x): #thruster rate limit
        delta_z = x[0:3]
        u_rate_max = np.array([
            thruster.u_rate * dt/1000
            for thruster in self.thrusters[:3]
        ])
        return u_rate_max - np.abs(delta_z)
    def wrench_constraint(x):
        delta_z = x[0:3]
        delta_alpha = x[3:5]
        s = x[5:8]
        z = u_now_KN + delta_z
        alpha = np.array([alpha_now[0],alpha_now[1] + delta_alpha[0],alpha_now[2] + delta_alpha[1]])

        return B_matrix(alpha, self) @ z - tau_d[[0, 1, 5]] / 1000 - s

    constraints = [
    {"type": "ineq","fun": thrust_constraint},
    {"type": "eq","fun": wrench_constraint},
    {"type": "ineq","fun": azimuth_rate_constraint},
    {"type": "ineq","fun": thrust_rate_constraint}
    ]
    #intial guess
    x0=np.zeros(8)
    result=minimize(cost, x0, method='SLSQP', constraints=constraints, options={
            'ftol': 1e-8,
            'maxiter': 2000,
            'disp': False
        })

    if result.success: #solution found, check if feasible
        res=result.x
        # x:[delta_z, delta_alpha, s]
        tau_achieved = B_matrix(np.array([alpha_now[0],alpha_now[1] + res[3],alpha_now[2] + res[4]]), self) @ (u_now_KN + res[0:3]) * 1000
        u_cmd =(u_now_KN + res[0:3])*1000

        #wrap angles to [-pi, pi]
        alpha_cmd = np.array([alpha_now[0], alpha_now[1] + res[3], alpha_now[2] + res[4]])
        alpha_cmd = np.arctan2(np.sin(alpha_cmd), np.cos(alpha_cmd))
        if feasible(u_cmd, 1e-6, self):
            return u_cmd, alpha_cmd, tau_achieved, True #solution found and feasible
        else:
            return u_cmd, alpha_cmd, tau_achieved, False

    #failed optimization, return zeros
    return np.zeros(3), np.zeros(3), np.zeros(3), False

def allocate_part_1(tau_d, u_now, alpha_now, self):
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
        return np.zeros(self.n), np.zeros(self.n)

    #weights for the cost function:
    Q=np.diag([100*1e6,100*1e6,100*1e6]) #error penalty
    #Q=np.diag([100,100,100]) #error penalty
    R = np.diag([
        1.0 / self.thrusters[0].u_max**2,
        1.0 / self.thrusters[1].u_max**2,
        1.0 / self.thrusters[1].u_max**2,
        1.0 / self.thrusters[2].u_max**2,
        1.0 / self.thrusters[2].u_max**2,
    ])

    Fx1_now = u_now[1] * np.cos(alpha_now[1])
    Fy1_now = u_now[1] * np.sin(alpha_now[1])
    Fx2_now = u_now[2] * np.cos(alpha_now[2])
    Fy2_now = u_now[2] * np.sin(alpha_now[2])

    z0 = np.array([u_now[0], Fx1_now, Fy1_now, Fx2_now, Fy2_now])/1000

    #building the cost function:
    def cost(z): #3x5 5x1 - 3x1
        s = B_e @ z - tau_d[[0,1,5]]/1000 #error
        return s.T @ Q @ s + (z-z0).T @ R @ (z-z0)

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
    z=result.x
    if result.success:
        if feasible(z, 1e-6, self): #solution is found and feasible
            model=1
            tau_achieved = B_e @ z * 1000
            u_cmd, alpha_cmd = recover_thrust(z)
            return u_cmd, alpha_cmd, B_e, tau_achieved, model

    #no solution found or infeasible solution
    model=0
    z=saturation_mode(B_e, tau_d, self) #checks feasibility, returns zero if no solution is found
    tau_achieved = B_e @ z * 1000
    u_cmd, alpha_cmd = recover_thrust(z)
    return u_cmd, alpha_cmd, B_e, tau_achieved, model



def dynamics(u_now, alpha_now, u_cmd, alpha_cmd,dt ,self,t, model): #returns true if rate limits are passed
    #check if rotation rate is ok:
    if self.thrusters[1].rot_speed<np.abs((alpha_cmd[1]-alpha_now[1])/dt) or self.thrusters[2].rot_speed<np.abs((alpha_cmd[2]-alpha_now[2])/dt):
        print(f"Azimuth rotation speed exceeded at time {t} model {model}, delta alpha1: {np.abs((alpha_cmd[1]-alpha_now[1])/dt)}, delta alpha2: {np.abs((alpha_cmd[2]-alpha_now[2])/dt)}")
        return False
    #thruster rate limit check:
    if self.thrusters[0].u_rate<np.abs((u_cmd[0]-u_now[0])/dt) or self.thrusters[1].u_rate<np.abs((u_cmd[1]-u_now[1])/dt) or self.thrusters[2].u_rate<np.abs((u_cmd[2]-u_now[2])/dt):
        print(f"Thruster rate limit exceeded at time {t}, model {model}")
        return False
    return True

def saturation_mode(B_e, tau_d, self): #infeasible demand gets called if part 1 fails to find a solution
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

    if result.success:
        z = result.x[:5]
        k = result.x[5]
        if feasible(z,1e-3, self):
            #print("Warning: Saturated thrust!")
            return z
    
    print("Warning: Zero Thrust!") #no solution found
    return np.zeros(5)

def feasible(z, limit, self): #returns True if the solution is feasible
    #no azimuth transformation part 2
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

def recover_thrust(z): #part 1 method
    z=1000*z #back to kN
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