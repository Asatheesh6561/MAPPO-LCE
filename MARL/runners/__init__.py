REGISTRY = {}

from .parallel_runner import ParallelRunner
from .episode_runner import EpisodeRunner

REGISTRY["parallel"] = ParallelRunner
REGISTRY["episode"] = EpisodeRunner
