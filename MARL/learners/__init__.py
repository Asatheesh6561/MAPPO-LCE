from .local_ppo_learner import LocalPPOLearner
from .maddpg_learner import MADDPGLearner
from .actor_critic_learner import ActorCriticLearner
from .cppo_learner import CPPOLearner
from .qtran_learner import QLearner

REGISTRY = {}

REGISTRY["local_ppo_learner"] = LocalPPOLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["cppo_learner"] = CPPOLearner
REGISTRY["qtran_learner"] = QLearner
