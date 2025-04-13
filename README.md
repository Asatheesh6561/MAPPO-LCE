# CMARL-Traffic

Source code of "[A Constrained Multi-Agent Reinforcement Learning Approach to Autonomous Traffic Signal Control](https://arxiv.org/abs/2503.23626)", which proposes a new method for constrained MARL on Adaptive Traffic Signal Control (ATSC), Multi-Agent Proximal Policy Optimization with Lagrange Cost Estimator (MAPPO-LCE).

# Usage
We use [CityFlow](https://github.com/zyr17/CityFlow) as traffic simulator.

To run our code, use python 3.9.20 in a conda environment.

## Build
``` bash
conda create -n mappolce python=3.9
conda activate mappolce
git clone git@github.com:Asatheesh6561/MAPPO-LCE.git
cd MAPPO-LCE & pip install -r requirements.txt
git clone git@github.com:zyr17/CityFlow.git
cd CityFlow && pip install . --upgrade && cd ..
conda install -c conda-forge libstdcxx-ng
```

## Run

As an example, to run MAPPO on the HZ environment with the PhaseSkip constraint:
```bash
python main.py --cityflow-config=configs/cityflow/HZ.yml --config=MARL/configs/algs/mappo.yaml --constraint=PhaseSkip
```

Run different algorithms by substituting the corresponding .yml files in the arguments. The constraint choices and all other arguments are listed in `Environment/utils/arguments.py`.

## Environment

The definition of the gym environment is in `Environment/cityflow_env.py`. It runs in another 
process managed by the environment run files in `MARL/run`.

## Dataset

We provide the HZ, JN, and NY datasets in the `data.tgz` tarball. You can also find the original dataset files on the [COLight](https://github.com/wingsweihua/colight) github page. To use them, run
```
tar -zxvf data.tgz -C Environment
```

## Arguments and Configs

We parse arguments by `Environment/utils/argument.py`. The config for CityFlow and main
process are two separate configs. All CityFlow configs are stored in 
`configs/cityflow`, and algorithm hyperparameters are stored in `MARL/configs`.

## Logs

We use Weights and Biases for logging, but this can be disabled by setting `--enable-wandb=False`.
If you want to enable Weights and Biases, please set the default wandb entity and project name in `Environments/utils/arguments.py`.

## BibTeX Citation

If you use MAPPO-LCE in your work, we would appreciate including the following citation:
```
@misc{satheesh2025constrainedmultiagentreinforcementlearning,
      title={A Constrained Multi-Agent Reinforcement Learning Approach to Autonomous Traffic Signal Control}, 
      author={Anirudh Satheesh and Keenan Powell},
      year={2025},
      eprint={2503.23626},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2503.23626}, 
}
```
