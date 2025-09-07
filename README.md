# Meta-RL Baseline

This repository provides a baseline for meta-reinforcement learning algorithms, modernized to be compatible with the latest versions of standard reinforcement learning libraries.

### Algorithms
*   **MAML-TRPO** 

### Environments
*   **MuJoCo:** `HalfCheetahForwardBackward-v5`
*   **Meta-World ML1**

### Future Work (To-Do)
*   Implement the **RL<sup>2</sup>** algorithm.
*   Expand support for more `mujoco` and `metaworld` environments.
*   Add custom environments.

## Credits

This codebase is heavily based on the [learn2learn](https://github.com/learnables/learn2learn/) library. The original structure and algorithms were adapted from their repository.

The primary purpose of this repository is to update the dependencies and ensure compatibility with the current Python and RL ecosystema and add new algorithms and environments.

## Key Updates & Modernization

The key migrations include:
*   **Environment API:** Migrated from `gym==0.23.0` to `gymnasium==1.2.0`. Gymnasium is the official and maintained fork of OpenAI's Gym library.
*   **Physics Engine:** Updated from the deprecated `mujoco-py==2.1.2.14` to the officially supported `mujoco==3.3.5` developed by Google DeepMind.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KnockType/meta-rl-baseline.git
    cd meta-rl-baseline
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is provided for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run a training script, execute the main Python file. For example, to run the MAML-TRPO implementation:
```bash
python maml_trpo.py
```