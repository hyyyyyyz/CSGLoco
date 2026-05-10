"""Scripted get-up controller + safety gate for SafeRecovery evaluation.

Non-RL baseline: state machine with pre-defined joint trajectories.
Actions are in the policy's action space (offsets from default, scaled by action_scale=0.25).
So action=4.0 → 1.0 rad offset from default joint angle.
"""
import torch


class ScriptedRecoveryController:

    def __init__(self, num_envs, device, action_scale=0.25,
                 max_action=12.0, phase_duration=30):
        self.num_envs = num_envs
        self.device = device
        self.action_scale = action_scale
        self.max_action = max_action  # ±3.0 rad in joint space
        self.phase_duration = phase_duration

        self.fallen_step_count = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.is_fallen_internal = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.fallen_step_count.zero_()
            self.is_fallen_internal.zero_()
        else:
            self.fallen_step_count[env_ids] = 0
            self.is_fallen_internal[env_ids] = False

    def get_actions(self, obs):
        N = obs.shape[0]
        actions = torch.zeros(N, 12, device=self.device)

        grav_x = obs[:, 6]
        grav_y = obs[:, 7]
        grav_z = obs[:, 8]

        upright = grav_z < -0.5
        on_back = grav_z > 0.3
        on_side = (torch.abs(grav_x) > 0.6) | (torch.abs(grav_y) > 0.6)
        fallen = ~upright

        newly_fallen = fallen & (~self.is_fallen_internal)
        self.fallen_step_count[newly_fallen] = 0
        self.is_fallen_internal[fallen] = True
        self.is_fallen_internal[upright] = False
        self.fallen_step_count[fallen] += 1

        pd = self.phase_duration
        step = self.fallen_step_count

        # Joint indices: [hip, thigh, calf] x [FL, FR, RL, RR]
        # hip: 0,3,6,9  thigh: 1,4,7,10  calf: 2,5,8,11
        THIGHS = [1, 4, 7, 10]
        CALVES = [2, 5, 8, 11]
        HIPS = [0, 3, 6, 9]

        # --- ON BACK ---
        # Phase 1 (0-pd): Aggressive tuck — pull legs tight
        p1_back = on_back & (step <= pd)
        if p1_back.any():
            for j in THIGHS:
                actions[p1_back, j] = -8.0  # strong tuck (target ~default-2.0)
            for j in CALVES:
                actions[p1_back, j] = 6.0   # fold calf up

        # Phase 2 (pd-2pd): Asymmetric push to roll over
        p2_back = on_back & (step > pd) & (step <= 2 * pd)
        if p2_back.any():
            # Push hard with left legs to roll right
            actions[p2_back, 0] = 8.0    # FL hip out
            actions[p2_back, 1] = 10.0   # FL thigh extend strongly
            actions[p2_back, 2] = -4.0   # FL calf extend
            actions[p2_back, 6] = 8.0    # RL hip out
            actions[p2_back, 7] = 10.0   # RL thigh extend
            actions[p2_back, 8] = -4.0   # RL calf extend
            # Tuck right legs
            actions[p2_back, 3] = -4.0   # FR hip in
            actions[p2_back, 4] = -6.0   # FR thigh tuck
            actions[p2_back, 9] = -4.0   # RR hip in
            actions[p2_back, 10] = -6.0  # RR thigh tuck

        # Phase 3 (2pd-3pd): Push up — all legs extend
        p3_back = on_back & (step > 2 * pd) & (step <= 3 * pd)
        if p3_back.any():
            for j in THIGHS:
                actions[p3_back, j] = 6.0   # extend thighs
            for j in CALVES:
                actions[p3_back, j] = -4.0  # straighten calves

        # --- ON SIDE ---
        p_side = on_side & (~on_back)
        if p_side.any():
            phase = step[p_side] % (3 * pd)
            idx_side = p_side.nonzero(as_tuple=False).squeeze(-1)
            gx = grav_x[idx_side]
            left_down = gx > 0

            # Phase 1: tuck
            m1 = phase <= pd
            idx_1 = idx_side[m1]
            if len(idx_1) > 0:
                for j in THIGHS:
                    actions[idx_1, j] = -6.0
                for j in CALVES:
                    actions[idx_1, j] = 4.0

            # Phase 2: push with downside legs
            m2 = (phase > pd) & (phase <= 2 * pd)
            idx_2 = idx_side[m2]
            ld2 = left_down[m2]
            if len(idx_2) > 0:
                for i, ei in enumerate(idx_2):
                    if ld2[i]:  # left down, push with left
                        actions[ei, 1] = 10.0; actions[ei, 7] = 10.0
                        actions[ei, 2] = -4.0; actions[ei, 8] = -4.0
                    else:  # right down
                        actions[ei, 4] = 10.0; actions[ei, 10] = 10.0
                        actions[ei, 5] = -4.0; actions[ei, 11] = -4.0

            # Phase 3: all extend
            m3 = phase > 2 * pd
            idx_3 = idx_side[m3]
            if len(idx_3) > 0:
                for j in THIGHS:
                    actions[idx_3, j] = 6.0
                for j in CALVES:
                    actions[idx_3, j] = -4.0

        # --- FACE DOWN ---
        face_down = fallen & (~on_back) & (~on_side)
        if face_down.any():
            for j in THIGHS:
                actions[face_down, j] = 8.0   # strong push up
            for j in CALVES:
                actions[face_down, j] = -4.0

        # Phase 4+: hold default standing
        hold = fallen & (step > 3 * pd)
        if hold.any():
            actions[hold] = 0.0

        # Safety gate: clamp
        actions = torch.clamp(actions, -self.max_action, self.max_action)
        return actions
