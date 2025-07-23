# entorno.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RealisticHomeCareEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    WORKING_DAY_END = 18 * 60

    def __init__(self, df, equipos_bases, team_types, n_teams=3, max_possible_visits_per_day=100, seed=42):
        super().__init__()
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.df = df.reset_index(drop=True)
        self.n_patients = len(df)
        self.n_teams = n_teams
        self.equipos_bases = {tid: np.array(pos) for tid, pos in equipos_bases.items()}
        self.team_types = team_types
        self.max_steps = max_possible_visits_per_day

        low_eq = [0.0, 0.0, 0.0] * self.n_teams
        high_eq = [10.0, 10.0, float(self.WORKING_DAY_END + 200)] * self.n_teams
        low_pat = [0.0, 0.0, 0.0, 0.0, 0.0] * self.n_patients
        high_pat = [10.0, 10.0, 1.0, float(24*60), float(24*60)] * self.n_patients

        low = np.array(low_eq + low_pat, dtype=np.float32)
        high = np.array(high_eq + high_pat, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_teams * self.n_patients)
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        sorted_team_ids = sorted(self.equipos_bases.keys())
        self.equipos_status = {
            tid: {"pos": self.equipos_bases[tid].copy(), "time": 8 * 60, "visits": []}
            for tid in sorted_team_ids
        }
        self.patients_status = []
        for idx, row in self.df.iterrows():
            w_ini = min(int(row["window_start"][:2]) * 60 + int(row["window_start"][3:5]), 1439)
            w_end = min(int(row["window_end"][:2]) * 60 + int(row["window_end"][3:5]), 1439)
            self.patients_status.append({
                "patient_id": row["patient_id"],
                "pos": np.array([row["lat"], row["lon"]]),
                "attended": 0,
                "window_start": w_ini,
                "window_end": w_end,
                "estimated_care_time": row["estimated_care_time"],
                "required_team_type": row["required_team_type"],
                "actual_arrival_time": -1
            })
        self.steps_taken = 0
        self.total_reward = 0
        self.invalid_action_count = 0
        self.current_day_itinerary = []
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        equipos_flat = []
        for tid in sorted(self.equipos_bases.keys()):
            equipos_flat.extend(self.equipos_status[tid]["pos"].tolist())
            equipos_flat.append(self.equipos_status[tid]["time"])
        patients_flat = []
        for pat in self.patients_status:
            patients_flat.extend(pat["pos"].tolist())
            patients_flat.append(pat["attended"])
            patients_flat.append(pat["window_start"])
            patients_flat.append(pat["window_end"])
        return np.array(equipos_flat + patients_flat, dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        truncated = False

        team_idx = action // self.n_patients
        patient_idx = action % self.n_patients

        sorted_team_ids = sorted(self.equipos_bases.keys())
        team_id = sorted_team_ids[team_idx]
        team_current_pos = self.equipos_status[team_id]["pos"]
        team_current_time = self.equipos_status[team_id]["time"]
        team_type = self.team_types[team_id]

        patient_info = self.patients_status[patient_idx]
        patient_pos = patient_info["pos"]
        patient_attended = patient_info["attended"]
        patient_w_start = patient_info["window_start"]
        patient_w_end = patient_info["window_end"]
        patient_care_time = patient_info["estimated_care_time"]
        patient_required_type = patient_info["required_team_type"]

        is_invalid_action = False

        if patient_attended == 1:
            reward += -40
            is_invalid_action = True
        if patient_required_type not in team_type:
            reward += -40
            is_invalid_action = True

        self.steps_taken += 1

        if is_invalid_action:
            self.invalid_action_count += 1
        else:
            self.invalid_action_count = 0

        if not is_invalid_action:
            dist = np.linalg.norm(team_current_pos - patient_pos) * 10
            travel_time = int(dist * 5)
            arrival_time_at_patient = team_current_time + travel_time

            late_penalty = 0
            wait_time_penalty = 0
            bonus_in_window = 0
            bonus_any_visit = 0

            if arrival_time_at_patient > patient_w_end:
                retraso = arrival_time_at_patient - patient_w_end
                late_penalty = retraso
            if arrival_time_at_patient < patient_w_start:
                wait_time = patient_w_start - arrival_time_at_patient
                arrival_time_at_patient = patient_w_start
                wait_time_penalty = wait_time * 0.1

            visit_end_time = arrival_time_at_patient + patient_care_time

            overtime_penalty = 0
            if visit_end_time > self.WORKING_DAY_END:
                overtime = visit_end_time - self.WORKING_DAY_END
                overtime_penalty = overtime * 2

            self.equipos_status[team_id]["pos"] = patient_pos.copy()
            self.equipos_status[team_id]["time"] = visit_end_time
            self.patients_status[patient_idx]["attended"] = 1
            self.patients_status[patient_idx]["actual_arrival_time"] = arrival_time_at_patient

            if patient_w_start <= arrival_time_at_patient <= patient_w_end:
                bonus_in_window = 60
            bonus_any_visit = 40

            self.current_day_itinerary.append({
                "day": self.df.iloc[patient_idx]["day"],
                "patient_id": patient_info["patient_id"],
                "team_id": team_id,
                "arrival_time": arrival_time_at_patient,
                "end_time": visit_end_time,
                "travel_time": travel_time,
                "care_time": patient_care_time,
                "window_start": patient_w_start,
                "window_end": patient_w_end,
                "is_in_window": (patient_w_start <= arrival_time_at_patient <= patient_w_end)
            })

            reward += bonus_any_visit
            reward += bonus_in_window
            reward -= travel_time * 0.3
            reward -= late_penalty
            reward -= wait_time_penalty
            reward -= overtime_penalty

        self.total_reward += reward

        all_patients_attended = all(p["attended"] == 1 for p in self.patients_status)
        if all_patients_attended:
            done = True

        if self.steps_taken >= self.max_steps:
            done = True
            truncated = True
            unattended_count = sum(1 for p in self.patients_status if p["attended"] == 0)
            reward -= unattended_count * 20

        if self.invalid_action_count > self.n_patients * self.n_teams:
            done = True
            truncated = True

        if done:
            final_overtime_penalty = 0
            for tid in sorted(self.equipos_bases.keys()):
                team_final_pos = self.equipos_status[tid]["pos"]
                team_final_time = self.equipos_status[tid]["time"]
                base_pos = self.equipos_bases[tid]
                return_dist = np.linalg.norm(team_final_pos - base_pos) * 10
                return_travel_time = int(return_dist * 5)
                arrival_at_base_time = team_final_time + return_travel_time
                if arrival_at_base_time > self.WORKING_DAY_END:
                    final_overtime = arrival_at_base_time - self.WORKING_DAY_END
                    final_overtime_penalty += final_overtime * 5
            reward -= final_overtime_penalty
            self.total_reward += -final_overtime_penalty

        obs = self._get_obs()
        info = {
            "total_reward": self.total_reward,
            "current_itinerary": self.current_day_itinerary
        }
        return obs, reward, done, truncated, info
