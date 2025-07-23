# evaluacion.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from entorno import RealisticHomeCareEnv
from stable_baselines3 import DQN

# Cargar datos y diccionarios (igual que en entrenamiento.py)
GLOBAL_SEED = 42
N_TEAMS = 3
N_PACIENTES = 8
N_DIAS = 20

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

# Cargar modelo entrenado
model = DQN.load("modelo_rl_hhc")

estadistica = []
itinerarios_completos = []

for test_day in range(1, N_DIAS + 1):
    df_day = data_all[data_all["day"] == test_day].copy().reset_index(drop=True)
    if len(df_day) != N_PACIENTES:
        print(f"Skipping day {test_day} due to data mismatch.")
        continue

    env_eval = RealisticHomeCareEnv(
        df_day,
        team_dict,
        team_type_dict,
        n_teams=N_TEAMS,
        max_possible_visits_per_day=N_TEAMS * N_PACIENTES * 2,
        seed=GLOBAL_SEED
    )
    obs, info_reset = env_eval.reset()
    done = False
    truncated = False
    episode_reward = 0
    day_visits_info = []
    team_visit_order_counter = defaultdict(int)

    while not done and not truncated:
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs, dtype=np.float32)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_eval.step(action)
        episode_reward += reward
        if len(info["current_itinerary"]) > len(day_visits_info):
            last_visit = info["current_itinerary"][-1]
            team_id = last_visit["team_id"]
            team_visit_order_counter[team_id] += 1
            last_visit["order_in_day_for_team"] = team_visit_order_counter[team_id]
            day_visits_info.append(last_visit)
    itinerarios_completos.extend(day_visits_info)
    ok_visits = sum(1 for v in day_visits_info if v["is_in_window"])
    total_successful_visits = len(day_visits_info)
    estadistica.append({
        "dia": test_day,
        "recompensa_total": episode_reward,
        "visitas_cumplidas_ventana": ok_visits,
        "total_visitas_realizadas": total_successful_visits,
        "pacientes_atendidos_final": sum(p["attended"] == 1 for p in env_eval.patients_status),
        "total_pacientes_dia": env_eval.n_patients
    })

df_stats = pd.DataFrame(estadistica)
print("\n--- Resumen de Cumplimiento por Día ---")
print(df_stats)
total_cumplidas_general = df_stats["visitas_cumplidas_ventana"].sum()
total_realizadas_general = df_stats["total_visitas_realizadas"].sum()
porcentaje_cumplimiento = (total_cumplidas_general / total_realizadas_general * 100) if total_realizadas_general > 0 else 0
print(f"\nPromedio general de visitas en ventana horaria: {total_cumplidas_general} / {total_realizadas_general} = {porcentaje_cumplimiento:.1f}%")
print(f"Recompensa promedio por episodio: {df_stats['recompensa_total'].mean():.2f}")

plt.figure(figsize=(10, 5))
plt.bar(df_stats["dia"], df_stats["visitas_cumplidas_ventana"], color="green", label="Visitas en Ventana")
plt.bar(df_stats["dia"], df_stats["total_visitas_realizadas"] - df_stats["visitas_cumplidas_ventana"],
        bottom=df_stats["visitas_cumplidas_ventana"], color="orange", label="Visitas Fuera de Ventana")
plt.plot(df_stats["dia"], df_stats["total_pacientes_dia"], "--", color="blue", label="Total Pacientes Dia")
plt.xlabel("Día")
plt.ylabel("Número de Visitas")
plt.title("Rendimiento del Agente por Día")
plt.legend()
plt.xticks(df_stats["dia"])
plt.tight_layout()
plt.show()

df_it_full = pd.DataFrame(itinerarios_completos)
if not df_it_full.empty:
    df_it_full["hora"] = df_it_full["arrival_time"].apply(lambda t: f"{int(t//60):02d}:{int(t%60):02d}")
    df_it_full["hora_fin"] = df_it_full["end_time"].apply(lambda t: f"{int(t//60):02d}:{int(t%60):02d}")
    df_it_full["cumplimiento"] = df_it_full["is_in_window"].replace({True: "En ventana", False: "Fuera de ventana"})
    df_it_full_sorted = df_it_full.sort_values(by=["day", "team_id", "order_in_day_for_team"])
    df_it_full_sorted["tipo_equipo"] = df_it_full_sorted["team_id"].map(team_type_dict)

    print("\nPrimeras 20 visitas detalladas de los itinerarios generados:")
    print(df_it_full_sorted[['day', 'order_in_day_for_team', 'team_id', 'patient_id', 'hora', 'hora_fin', 'cumplimiento', 'travel_time', 'care_time']].head(20).to_string())

    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df_it_full_sorted,
        x="team_id", hue="cumplimiento",
        palette={"En ventana": "green", "Fuera de ventana": "orange"}
    )
    plt.title("Cumplimiento de visitas por equipo")
    plt.xlabel("Equipo")
    plt.ylabel("Número de visitas")
    plt.legend(title="Cumplimiento")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df_it_full_sorted,
        x="tipo_equipo", hue="cumplimiento",
        palette={"En ventana": "green", "Fuera de ventana": "orange"}
    )
    plt.title("Cumplimiento de visitas por tipo de equipo")
    plt.xlabel("Tipo de Equipo")
    plt.ylabel("Número de visitas")
    plt.legend(title="Cumplimiento")
    plt.tight_layout()
    plt.show()

    # Exportar a Excel para reporte
    df_it_full_sorted.to_excel("itinerario_rl_hhc.xlsx", index=False)
    print("\nItinerario exportado a 'itinerario_rl_hhc.xlsx'")
else:
    print("No hay datos de itinerario para graficar.")
