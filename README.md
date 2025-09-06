# Meta-RL Baseline

This repository provides a baseline for meta-reinforcement learning algorithms, modernized to be compatible with the latest versions of standard reinforcement learning libraries. It is designed to be a starting point for researchers and practitioners looking to experiment with meta-RL techniques using up-to-date tools.

## Credits

This codebase is heavily based on the [learn2learn](https://github.com/learnables/learn2learn/) library. The original structure and algorithms were adapted from their repository.

The primary purpose of this repository is to update the dependencies and ensure compatibility with the current Python and RL ecosystem.

## Key Updates & Modernization

The core motivation behind this repository is to migrate from older, now-deprecated libraries to their modern successors. This makes the code easier to install, run, and maintain on current systems.

The key migrations include:
*   **Environment API:** Migrated from `gym==0.23.0` to `gymnasium==1.2.0`. Gymnasium is the official and maintained fork of OpenAI's Gym library.
*   **Physics Engine:** Updated from the deprecated `mujoco-py==2.1.2.14` to the officially supported `mujoco==3.3.5` developed by Google DeepMind.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/meta-rl-baseline.git
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
You can modify the parameters and configurations within the script to suit your experiments.

## Project Goals

*   To provide a stable, modern baseline for meta-RL research.
*   To simplify the setup process by using up-to-date and well-supported libraries.
*   To serve as an educational resource for those learning about meta-reinforcement learning with modern tools.

## Contributing

Contributions are welcome! If you find a bug or have an idea for an improvement, please open an issue or submit a pull request.