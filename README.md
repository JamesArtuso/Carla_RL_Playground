<!--
 * @Date: 2026-04-23
 * @Description: CARLA RL Playground README
-->

<div align="center">

<h1>CARLA RL Playground</h1>

<p>
A reinforcement learning playground for training and evaluating driving policies in CARLA using Gym-based environments and PPO / PPO-LSTM.
</p>

</div>

---

## Installation

**Recommended system: Ubuntu 20.04 or 22.04**

### 1. Local Installation

<details>
    <summary> Click to expand </summary>

#### Step 1: Setup conda environment
```bash
conda create -n gym-carla python=3.8
conda activate gym-carla
```

#### Step 2: Clone this repository
```bash
git clone https://github.com/JamesArtuso/Carla_RL_Playground.git
cd Carla_RL_Playground
```

#### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Install CARLA

Download **CARLA 0.9.13** and set the root directory:

```bash
export CARLA_ROOT=/path/to/your/carla
```

Add CARLA to your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```

(Optional) Add these to a script like:
```bash
source set_env_vars.sh
```

</details>

---

## Running CARLA

Start CARLA in a separate terminal:

```bash
cd $CARLA_ROOT
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000
```

---

## Training / Running

Run an experiment:

```bash
python run.py --agent_cfg ppo.yaml --env_cfg default.yaml
```

Example:

```bash
python run.py --agent_cfg ppo.yaml --env_cfg default.yaml --seed 0
```

---

## Project Structure

```text
carla_gym/
  env_configs/          Environment configs
  planning/config/      Agent configs (PPO, LSTM, etc.)
  gym_carla/            CARLA Gym environment
  planning/rl/     RL algorithms and workers

run.py                  Main training entry point
requirements.txt        Dependencies
```

---

## Notes

- Requires **CARLA 0.9.13** and **Python 3.8**
- CARLA must be running before executing `run.py`
- Training outputs (logs, models) are saved to the experiment directory

---

## Acknowledgements

Parts of this project are based on prior work using the MIT License.  
Please refer to included third-party license files for details.