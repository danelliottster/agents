import gin
import tensorflow as tf
from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common

@gin.configurable()
class NaiveEnsemble(tf_agent.TFAgent):
  """A Naive Ensemble which combines Q-funcitons via the selected method.

  Implements the DQN algorithm from

  "Human level control through deep reinforcement learning"
    Mnih et al., 2015
    https://deepmind.com/research/dqn/

  This agent also implements n-step updates. See "Rainbow: Combining
  Improvements in Deep Reinforcement Learning" by Hessel et al., 2017, for a
  discussion on its benefits: https://arxiv.org/abs/1710.02298

  Some notes:

  * Only uses each member's greedy policy.
  """

  def __init__(
    self,
    ensemble_members: list[tf_agent],
    time_step_spec: ts.TimeStep, # needs to match ensemble members
    action_spec: types.NestedTensorSpec, # needs to match ensemble members
    epsilon_greedy: Optional[types.FloatOrReturningFloat] = 0.1,
    n_step_update: int = 1,
    boltzmann_temperature: Optional[types.FloatOrReturningFloat] = None,
    # Params for debugging
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    train_step_counter: Optional[tf.Variable] = None,
    training_data_spec: Optional[types.NestedTensorSpec] = None,
    name: Optional[Text] = None 
    ) :
    
    tf.Module.__init__(self, name=name)
    self.ensemble_members = ensemble_members

    # intitialize members

    # setup policies: an exploitation and an exploration policy

    super(DqnAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=train_sequence_length,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        training_data_spec=training_data_spec,
    )



def _initialize(self) :
    """ Perform initialization step for each ensemble member. """
    for ensemble_member in self.ensemble_members :
        ensemble_member.initialize()

def _setup_policy(self) :
    pass

def _train(self , experience ) :
    """
    Just run the experiences through the members and perform and update.
    Return the losses (can TF.agents handle a list of losses?)
    Args:
      experience: The data points to use to update the ensemble member weights.  See super class for more details.

    Returns:
      TODO: Not sure how to deal with this here.
    """

def preprocess_sequence( self, experience ) :
    """
    Will add a feature to the supplied experiences to specify which ensemble member to send each data point to.
    """
    
    pass
