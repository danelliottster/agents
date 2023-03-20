import random

import numpy as np

import tensorflow_datasets as tfds

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

CORRUPTIONS = [ "translation" ]
ACTION_ORDER = [ "translation" ]
ACTION_SPEC_DIM = { "translation":2 }

def translate( img_in , translation ) :

    return np.roll( img_in , shift=translation , axis=(0,1) )

def create_empty_cumulative_actions() :

    cumulative_actions = {}
    for action_name , action_dim in ACTION_SPEC_DIM.items() :
        if action_dim > 1 :
            cumulative_actions[ action_name ] = [ 0 ] * action_dim
        else :
            cumulative_actions[ action_name ] = 0

class mnist_corrupted ( py_environment.PyEnvironment ) :

    def __init__( self , reward_fn , max_duration = 50 , selected_corruptions=["translation"] ) :
        """ Initialize the environment
        
        Args:
            reward_fn: a function which takes a MxM image numpy array and the desired label and returns the reward
            max_duration : maximum number of time step for each trial
        """

        self._selected_corruptions = selected_corruptions
        assert np.all( [ corruption in CORRUPTIONS for corruption in self._selected_corruptions ] ) , "One more more supplied corruption types are invalid."

        self._action_spec_dims = np.sum( [ ACTION_SPEC_DIM[ corruption ] for corruption in self._selected_corruptions ] )
        self._action_spec = array_spec.BoundedArraySpec( shape=( self._action_spec_dims , ) , dtype=np.int32 , minimum=-1 , 
                                                        maximum=1 , name='action' )
        self._observation_spec = array_spec.BoundedArraySpec( shape=( 28*28+len(self._action_spec_dims) , ) , dtype=np.int32 , 
                                                             minimum=None , maximum=None , name='observation' )
        self._step_count = 0    
        self._episode_ended = False
        self._max_duration = max_duration
        self._reward_fn = reward_fn

        self._mnist = tfds.load('mnist', split='train', as_supervised=True)
        self._data_iter = self._mnist.repeat().as_numpy_iterator() #this will create an iterator which iteratres forever

        self._state_img_orig = None
        self._state_img_label = None
        self._state_corruption_params = {}
        self._state_img_corrupted = None
        self._state_cumulative_actions = create_empty_cumulative_actions()

        super(mnist_corrupted, self).__init__()

    def sample_translation( self ) :

        return [ random.choice( list( range(5) ) ) , random.choice( list( range(5) ) ) ]
    
    def action_spec( self ):
        return self._action_spec

    def observation_spec( self ):
        return self._observation_spec
    
    def pack_cumulative_actions( self ) :
        
        cumulative_action_state = []
        for action_name in ACTION_ORDER :
            if action_name in self._selected_corruptions :
                if type( self._state_cumulative_actions[ action_name ] ) is list :
                    cumulative_action_state += self._state_cumulative_actions[ action_name ]
                else :
                    cumulative_action_state += [ self._state_cumulative_actions[ action_name ] ]
        return cumulative_action_state
    
    def unpack_actions( self , action_vector ) :

        idx = 0
        unpacked_action = {}
        for action_name in ACTION_ORDER :
            if action_name in self._selected_corruptions :
                if type( self._state_cumulative_actions[ action_name ] ) is list :
                    tmp_size = len( self._state_cumulative_actions[ action_name ] )
                    unpacked_action[ action_name ] = [0] * tmp_size
                    for tmp_idx in range( tmp_size ) :
                        unpacked_action[ action_name ][ tmp_idx ] = action_vector[ idx , 0 ]
                else :
                    unpacked_action[ action_name ] = action_vector[ idx , 0 ]

    def accumulate_action ( self , unpacked_actions ) :

        for action_name , action_value in unpacked_actions.items() :
            self._state_cumulative_actions[ action_name ] += action_value
    
    def pack_state( self ) :

        dimension = self._observation_spec.num_values + self._action_spec.num_values
        state_vector = np.zeros( ( dimension , 1 ) )
        state_vector[ 0:self._observation_spec.num_values , 0 ] = self._state_img_corrupted.flatten()
        state_vector[ self._observation_spec.num_values: , 0 ] = self.pack_cumulative_actions()
        return state_vector

    def _reset(self):
        """ Generate a new, randomly-corrupted MNIST image. """

        # select a MNIST image
        self._state_img_orig , self._state_img_label = next( self._data_iter )

        ###### corrupt it! ######

        self._state_corruption_params = {}

        # translation
        self._state_corruption_params[ "translation" ] = self.sample_translation()
        self._state_img_corrupted = translate( self._state_img_orig , self._state_corruption_params[ "translation" ] )

        ###### end of corruptions ######

        self._episode_ended = False
        self._step_count = 0

        return ts.restart( self.pack_state() )

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # incorporate action into cumulated action
        unpacked_action = self.unpack_actions( action )
        self.accumulate_action( unpacked_action )

        # create observation
        observation = self.pack_state()

        # determine reward
        reward = self._reward_fn( observation , self._state_img_label )

        # done, send back the observation and reward
        # object type determined by whether or not this is the final time step
        self._step_count += 1
        if self._step_count >= self._max_duration :
            self._episode_ended = True  # will trigger reset next step
            time_step = ts.termination( observation, reward )
        else :
            time_step = ts.transition( observation , reward )

        return time_step
