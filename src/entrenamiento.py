# entrenamiento.py

import numpy as np
import pandas as pd
from entorno import RealisticHomeCareEnv
from stable_baselines3 import DQN

# Semilla global para reproducibilidad
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Parámetros
N_TEAMS = 3
N_PACIENTES = 8
N_DIAS = 20

# Configuración equipos
teams_info = pd.DataFrame({
    "team_id": [f"E{i+1}" for i in range(N_TEAMS)],
    "team_type": ["A", "B", "A"],
    "base_lat": [2, 7, 4],
    "base_lon": [2, 8, 6]
})

def simular_pacientes(ndias=N_DIAS, npac=N_PACIENTES):
    rows = []
    for d in range(ndias):
        for i in range(npac):
            lat = np.random.uniform(1, 9)
            lon = np.random.uniform(1, 9)
            start = np.random.randint(8, 15)
            end = start + np.random.randint(2, 4)
            care_time = np.random.randint(25, 50)
            tipo = np.random.choice(["A", "B"])
            rows.append({
                "day": d+1,
                "patient_id": f"P{i+1}_Day{d+1}",
                "lat": lat,
                "lon": lon,
                "window_start": f"{start:02d}:00",
                "window_end": f"{end:02d}:00",
                "estimated_care_time": care_time,
                "required_team_type": tipo
            })
    return pd.DataFrame(rows)

data_all = simular_pacientes()

team_dict = teams_info.set_index("team_id")[["base_lat", "base_lon"]].T.to_dict("list")
team_type_dict = teams_info.set_index("team_id")["team_type"].to_dict()

# ENTRENAMIENTO
env_train = RealisticHomeCareEnv(
    data_all,
    team_dict,
    team_type_dict,
    n_teams=N_TEAMS,
    max_possible_visits_per_day=N_TEAMS * N_PACIENTES * 2,
    seed=GLOBAL_SEED
)

model = DQN(
    "MlpPolicy",
    env_train,
    verbose=1,
    buffer_size=200_000,
    learning_starts=2_000,
    batch_size=128,
    train_freq=(4, "step"),
    target_update_interval=1_000,
    gamma=0.995,
    exploration_fraction=0.5,
    exploration_final_eps=0.02,
    learning_rate=3e-4,
    tensorboard_log="./tb_hhc_rl/"
)

model.learn(total_timesteps=1_000_000)
model.save("modelo_rl_hhc")
print("Entrenamiento RL finalizado y modelo guardado.")
