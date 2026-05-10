"""Simplified MPC Controller for Quadruped Locomotion

A QP-based MPC for baseline comparison. Uses simplified dynamics and
quadratic programming for real-time control.
"""
import numpy as np
import torch

class QuadrupedMPC:
    """Simplified MPC controller for quadruped locomotion.

    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    Control: [f_x, f_y, f_z] for each foot (4 feet = 12D control)
    """

    def __init__(self, dt=0.02, horizon=10, mass=12.0, gravity=9.81):
        self.dt = dt
        self.horizon = horizon
        self.mass = mass
        self.gravity = gravity

        # Robot parameters (A1)
        self.foot_positions_body = np.array([
            [0.1805, 0.047, 0.0],   # FR
            [0.1805, -0.047, 0.0],  # FL
            [-0.1805, 0.047, 0.0],  # RR
            [-0.1805, -0.047, 0.0]  # RL
        ])

        # Inertia (simplified, diagonal)
        self.inertia = np.diag([0.035, 0.12, 0.1])

        # Constraints
        self.mu = 0.6  # friction coefficient
        self.fz_min = 10.0  # minimum normal force
        self.fz_max = 200.0  # maximum normal force

        # Reference tracking weights
        self.Q_pos = np.diag([100, 100, 100])  # position tracking
        self.Q_vel = np.diag([10, 10, 10])     # velocity tracking
        self.Q_ori = np.diag([50, 50, 10])     # orientation tracking
        self.R = np.diag([0.01] * 12)          # control regularization

        # Try to import CasADi for QP solving
        try:
            import casadi as ca
            self.has_casadi = True
        except ImportError:
            print("Warning: CasADi not available, using simplified controller")
            self.has_casadi = False

    def rotation_matrix(self, roll, pitch, yaw):
        """Compute rotation matrix from Euler angles."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R

    def compute_contact_schedule(self, phase):
        """Compute which feet are in contact based on gait phase."""
        # Simple trot gait
        contact = np.zeros(4)
        if phase < 0.5:
            contact[0] = 1  # FR
            contact[3] = 1  # RL
        else:
            contact[1] = 1  # FL
            contact[2] = 1  # RR
        return contact

    def solve_qp_simple(self, state, ref, contact):
        """Simplified QP solution (analytical, no optimization)."""
        # Extract state
        pos = state[0:3]
        ori = state[3:6]
        vel = state[6:9]
        omega = state[9:12]

        # Reference
        pos_ref = ref[0:3]
        vel_ref = ref[6:9]

        # PD control for position/velocity tracking
        kp_pos = 100.0
        kd_pos = 10.0
        acc_des = kp_pos * (pos_ref - pos) + kd_pos * (vel_ref - vel)

        # Add gravity compensation
        acc_des[2] += self.gravity

        # Distribute forces among contact feet
        num_contact = np.sum(contact)
        if num_contact == 0:
            return np.zeros(12)

        # Total desired force
        f_total = self.mass * acc_des

        # Distribute evenly among contact feet
        forces = np.zeros(12)
        for i in range(4):
            if contact[i] > 0.5:
                forces[3*i:3*i+3] = f_total / num_contact

                # Enforce friction cone (simplified)
                fx, fy, fz = forces[3*i:3*i+3]
                fz = np.clip(fz, self.fz_min, self.fz_max)
                f_horiz = np.sqrt(fx**2 + fy**2)
                if f_horiz > self.mu * fz:
                    scale = self.mu * fz / (f_horiz + 1e-6)
                    fx *= scale
                    fy *= scale
                forces[3*i:3*i+3] = [fx, fy, fz]

        return forces

    def solve_qp_casadi(self, state, ref, contact):
        """Solve QP using CasADi."""
        if not self.has_casadi:
            return self.solve_qp_simple(state, ref, contact)

        import casadi as ca

        # Decision variables: forces for each foot (12D)
        f = ca.MX.sym('f', 12)

        # Extract state
        pos = state[0:3]
        vel = state[6:9]

        # Reference tracking error
        pos_err = pos - ref[0:3]
        vel_err = vel - ref[6:9]

        # Predicted acceleration (simplified)
        total_force = ca.MX.zeros(3)
        for i in range(4):
            if contact[i] > 0.5:
                total_force += f[3*i:3*i+3]

        acc = total_force / self.mass - np.array([0, 0, self.gravity])

        # Cost function
        cost = 0
        # Tracking cost
        cost += ca.mtimes([pos_err.T, self.Q_pos, pos_err])
        cost += ca.mtimes([vel_err.T, self.Q_vel, vel_err])
        # Control regularization
        cost += ca.mtimes([f.T, self.R, f])

        # Constraints
        g = []
        lbg = []
        ubg = []

        for i in range(4):
            if contact[i] > 0.5:
                # Friction cone constraints
                fx, fy, fz = f[3*i], f[3*i+1], f[3*i+2]
                g.append(fz)
                lbg.append(self.fz_min)
                ubg.append(self.fz_max)

                g.append(fx - self.mu * fz)
                lbg.append(-ca.inf)
                ubg.append(0)

                g.append(-fx - self.mu * fz)
                lbg.append(-ca.inf)
                ubg.append(0)

                g.append(fy - self.mu * fz)
                lbg.append(-ca.inf)
                ubg.append(0)

                g.append(-fy - self.mu * fz)
                lbg.append(-ca.inf)
                ubg.append(0)
            else:
                # No contact: forces must be zero
                g.append(f[3*i:3*i+3])
                lbg.append([0, 0, 0])
                ubg.append([0, 0, 0])

        # Solve QP
        nlp = {'x': f, 'f': cost, 'g': ca.vertcat(*g)}
        solver = ca.nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        })

        # Initial guess
        f0 = np.zeros(12)

        try:
            sol = solver(x0=f0, lbg=lbg, ubg=ubg)
            forces = np.array(sol['x']).flatten()
        except:
            # Fallback to simple solution
            forces = self.solve_qp_simple(state, ref, contact)

        return forces

    def compute_control(self, state, ref, phase=0.0):
        """Compute control action.

        Args:
            state: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
            ref: reference state (same structure)
            phase: gait phase [0, 1]

        Returns:
            joint torques or foot forces
        """
        # Get contact schedule
        contact = self.compute_contact_schedule(phase)

        # Solve QP
        if self.has_casadi:
            forces = self.solve_qp_casadi(state, ref, contact)
        else:
            forces = self.solve_qp_simple(state, ref, contact)

        return forces


class MPCPolicyWrapper:
    """Wrapper to use MPC as a policy for legged_gym evaluation."""

    def __init__(self, device='cuda:0'):
        self.mpc = QuadrupedMPC(dt=0.02, horizon=10)
        self.device = device
        self.phase = 0.0

    def __call__(self, obs):
        """Forward pass compatible with RL policy interface."""
        # obs: [batch, obs_dim] from legged_gym
        # Convert to state representation

        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = obs

        batch_size = obs_np.shape[0]
        actions = np.zeros((batch_size, 12))

        for i in range(batch_size):
            # Extract state from observation (simplified mapping)
            # obs typically includes: [cmd_vel, projected_gravity, cmd_yaw, joint_pos, joint_vel, prev_action]
            # We'll approximate state from these

            state = self.obs_to_state(obs_np[i])

            # Reference: command velocity
            ref = np.zeros(12)
            # Assume command velocity is in first few dimensions
            ref[6:9] = obs_np[i, :3]  # cmd_vel

            # Compute control
            forces = self.mpc.compute_control(state, ref, self.phase)

            # Convert forces to joint torques (simplified)
            # This is a placeholder - real implementation needs Jacobian
            actions[i] = forces * 0.01  # Scale down for torque limits

        # Update phase
        self.phase = (self.phase + 0.1) % 1.0

        return torch.tensor(actions, dtype=torch.float32, device=self.device)

    def obs_to_state(self, obs):
        """Convert observation to state."""
        # Simplified: extract what we can from observation
        # This is approximate since obs doesn't contain full state

        state = np.zeros(12)
        # Approximate from projected gravity and commands
        state[0:3] = obs[3:6] * 0.3  # approximate position from gravity
        state[6:9] = obs[0:3]  # cmd_vel as approximate velocity
        return state


def test_mpc():
    """Test MPC controller."""
    mpc = QuadrupedMPC()

    # Test state
    state = np.array([0, 0, 0.3, 0, 0, 0, 0.5, 0, 0, 0, 0, 0])
    ref = np.array([0, 0, 0.3, 0, 0, 0, 0.5, 0, 0, 0, 0, 0])

    forces = mpc.compute_control(state, ref, phase=0.0)
    print("MPC Forces:", forces)
    print("MPC Force magnitude:", np.linalg.norm(forces))


if __name__ == "__main__":
    test_mpc()
