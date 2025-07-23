# RL-based Hospital-at-Home Route Optimization

This repository contains a proof-of-concept implementation of a Reinforcement Learning (RL) model to optimize daily itineraries for home health care (HHC) teams. The model is designed to assign visits to patients by clinical teams, considering spatial, temporal, and team composition constraints, and aims to maximize the percentage of visits performed within their scheduled time windows.

## Project Overview

- **Goal:** To create a data-driven tool for automatic and dynamic scheduling of hospital-at-home teams, leveraging RL for realistic, robust, and generalizable route planning.
- **Technology Readiness Level:** TRL 4 — validated in laboratory conditions with simulated data.
- **Main Components:**
    - Custom OpenAI Gym-compatible environment (`RealisticHomeCareEnv`) reflecting real-world constraints.
    - Simulation and visualization scripts for teams and patients.
    - RL training with Stable-Baselines3 DQN.
    - Performance metrics, route visualizations, and Excel export.

## Features

- Multi-team, multi-patient simulation with user-defined constraints.
- Realistic patient windows, care times, and compatibility by team type.
- Penalty and reward structure to promote feasible, timely itineraries.
- Output includes detailed itineraries and statistical summaries.

## How to Use

1. **Clone the repository**
    ```bash
    git clone https://github.com/mauricio-herrera/rl-hospital-at-home.git
    cd rl-hospital-at-home
    ```

2. **Set up your Python environment (recommended: Python 3.8–3.11)**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Run the Jupyter notebook**
    ```bash
    jupyter lab
    ```
    Open the notebook and execute the cells step by step.

4. **Results**
    - View performance plots directly in the notebook.
    - Itinerary Excel files are saved as `itinerario_rl_hhc.xlsx`.

## File Structure

- `rl_hhc_notebook.ipynb` — Main notebook with all code blocks.
- `entorno.py` — Environment definition (if used as a script/module).
- `entrenamiento.py` — Training script (optional if not using notebook).
- `evaluacion.py` — Evaluation and visualization (optional).
- `requirements.txt` — Python dependencies.
- `.gitignore` — Files/folders ignored by Git.

## Requirements

- Python 3.8+
- `gymnasium`, `stable-baselines3`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl` (for Excel export)
- See `requirements.txt` for full details.

## Results

The RL model achieves robust assignment of teams and generates detailed, feasible daily itineraries for all patients, with high compliance within care windows. Visualizations and exported reports are included for further analysis.

## Future Work

- Integrate real hospital data for advanced validation.
- Develop a web platform and mobile app for team access.
- Allow online retraining and scenario simulation for dispatchers.
- Compare with optimization and heuristic baselines.

## License

This project is released under the MIT License.

## Contact

For questions, please contact [M. Herrera](mailto:mherrera@udd.cl).

