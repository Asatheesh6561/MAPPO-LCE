from .parallel_run import run as parallel_run
from .episode_run import run as episode_run

REGISTRY = {}

REGISTRY["ippo_run"] = episode_run
REGISTRY["maddpg_run"] = episode_run
REGISTRY["mappo_run"] = episode_run
REGISTRY["maa2cc_run"] = episode_run
REGISTRY["mappolce_run"] = episode_run
REGISTRY["qtran_run"] = episode_run
