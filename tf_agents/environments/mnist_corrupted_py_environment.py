import random

import numpy as np
from abc import ABC , abstractmethod

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_datasets as tfds

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts



def create_empty_cumulative_actions() :
    """
    Create an empty dictionary of cumulative actions
    Returns:
        the empty dictionary
    """

    cumulative_actions = {}
    for action_name , action_dim in ACTION_SPEC_DIM.items() :
        if action_dim > 1 :
            cumulative_actions[ action_name ] = np.zeros( action_dim )
        else :
            cumulative_actions[ action_name ] = 0

    return cumulative_actions

def normalize_image( image_in , label_in ) :

    return tf.cast( image_in , tf.float32 ) / 255. , label_in


CORRUPTIONS = [ "translation" ]
ACTION_ORDER = [ "translation" ]
ACTION_FUNCTIONS = { "translation" : translate }
ACTION_SPEC_DIM = { "translation":2 }
REWARD_SCALAR = 100

class MnistCorrupted ( py_environment.PyEnvironment ) :

    def __init__( self , classifier , max_duration = 50 , selected_corruptions=["translation"] ) :
        """ Initialize the environment
        
        TODO:
            - modify reward function to penalize actions which do not improve the classification
            - add more corruptions
            - pad the image with zeros?
        Args:
            reward_fn: a function which takes a MxM image numpy array and the desired label and returns the reward
            max_duration : maximum number of time step for each trial
        """

        self._selected_corruptions = selected_corruptions
        assert np.all( [ corruption in CORRUPTIONS for corruption in self._selected_corruptions ] ) , "One more more supplied corruption types are invalid."

        self._action_spec_dims = np.sum( [ ACTION_SPEC_DIM[ corruption ] for corruption in self._selected_corruptions ] )
        self._action_spec = array_spec.BoundedArraySpec( shape=( ) , dtype=np.int32 , minimum= , maximum=1 , name='action' )
        self._observation_spec = array_spec.BoundedArraySpec( shape=( 28 , 28 ) , dtype=np.int32 , 
                                                             minimum=None , maximum=None , name='observation' )

        self._step_count = 0
        self._episode_ended = False
        self._max_duration = max_duration

        self._mnist = tfds.load('mnist', split='train', as_supervised=True)
        self._mnist = self._mnist.shuffle( 100 , reshuffle_each_iteration=True ).repeat()
        self._data_iter = self._mnist.as_numpy_iterator() #this will create an iterator which iteratres forever

        self._state_img_orig = None
        self._state_img_label = None
        self._state_img_classifier_label = None
        self._state_corruption_params = {}
        self._state_img_corrupted = None
        self._state_cumulative_actions = create_empty_cumulative_actions()

        self._classifier = classifier

        super(MnistCorrupted, self).__init__()

    def sample_translation( self ) :

        return np.array( [ random.choice( list( range(5) ) ) , random.choice( list( range(5) ) ) ] )
    
    def action_spec( self ):
        return self._action_spec

    def observation_spec( self ):
        return self._observation_spec
    
    # def pack_cumulative_actions( self ) :
        
    #     cumulative_action_state = []
    #     for action_name in ACTION_ORDER :
    #         if action_name in self._selected_corruptions :
    #             if type( self._state_cumulative_actions[ action_name ] ) is list :
    #                 cumulative_action_state += self._state_cumulative_actions[ action_name ]
    #             else :
    #                 cumulative_action_state += [ self._state_cumulative_actions[ action_name ] ]
    #     return cumulative_action_state

    def unpack_actions( self , action_vector ) :

        #TODO: check that the action vector matches the action spec

        idx = 0
        unpacked_action = {}
        for action_name in ACTION_ORDER :
            if action_name in self._selected_corruptions :
                if isinstance( self._state_cumulative_actions[ action_name ] , np.ndarray ) :
                    tmp_size = len( self._state_cumulative_actions[ action_name ] )
                    unpacked_action[ action_name ] = np.zeros( tmp_size )
                    for tmp_idx in range( tmp_size ) :
                        unpacked_action[ action_name ][ tmp_idx ] = action_vector[ idx ]
                else :
                    unpacked_action[ action_name ] = action_vector[ idx ]

        return unpacked_action

    def accumulate_action ( self , unpacked_actions ) :

        for action_name , action_value in unpacked_actions.items() :
            self._state_cumulative_actions[ action_name ] += action_value
            
    
    # def pack_state( self ) :

    #     dimension = self._observation_spec.num_values + self._action_spec.num_values
    #     state_vector = np.zeros( ( dimension , 1 ) )
    #     state_vector[ 0:self._observation_spec.num_values , 0 ] = self._state_img_corrupted.flatten()
    #     state_vector[ self._observation_spec.num_values: , 0 ] = self.pack_cumulative_actions()
    #     return state_vector
    
    def _reset(self):
        """ Generate a new, randomly-corrupted MNIST image. """

        # select a MNIST image
        self._state_img_orig , self._state_img_label = next( self._data_iter )

        # determine the class that the classifier thinks the uncorrupted image belongs to
        self._state_img_classifier_label = self._classifier.classify( self._state_img_orig )

        ###### corrupt it! ######

        self._state_corruption_params = {}

        # translation
        self._state_corruption_params[ "translation" ] = self.sample_translation()
        self._state_img_corrupted = translate( self._state_img_orig , self._state_corruption_params[ "translation" ] )

        ###### end of corruptions ######

        self._episode_ended = False
        self._step_count = 0

        # return ts.restart( self.pack_state() )
        return ts.restart( self._state_img_corrupted )
    
    def reward_function( self , observed_image ) :
        """
        Reward function for the environment.
        TOOD: 
            Add a penalty for each action taken.
        Args:
            observed_image: The image that the agent has observed.
        Returns:
            The reward for the current state.
        """
        # what does the classifier think the image is?
        class_inferences = self._classifier.classify( observed_image )
        # what is the probability that the classifier thinks the image is the correct class?
        correct_class_probability = class_inferences[ self._state_img_label ]

        return correct_class_probability * REWARD_SCALAR


    def _step( self , action ):
        """
        Apply the given action to the environment.
        Args:
            action: An action provided by the agent.
        Returns:          
            The next timestep.
        """

        # TODO: I'm not sure we need this.  Should be handled by step() of the parent class?
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # incorporate action into cumulated action
        unpacked_action = self.unpack_actions( action )
        self.accumulate_action( unpacked_action )

        # create observation
        observation = self.apply_cumulative_actions()

        # determine reward
        reward = self.reward_function( observation )

        # done, send back the observation and reward
        # object type determined by whether or not this is the final time step
        self._step_count += 1
        if self._step_count >= self._max_duration :
            self._episode_ended = True  # will trigger reset next step
            time_step = ts.termination( observation, reward )
        else :
            time_step = ts.transition( observation , reward )

        return time_step
    
    def apply_cumulative_actions( self ) :
        """ 
        Apply the cumulative actions to the original image.
        Returns the resulting observation.
        """

        # apply in order of ACTION_ORDER
        for action_name in ACTION_ORDER :

            observation = ACTION_FUNCTIONS[ action_name ]( self._state_img_corrupted , self._state_cumulative_actions[ "translation" ] )

        return observation
    
    def draw( self , fig_in=None ) :
        """
        Draw the current state of the environment.
        If fig_in is None, create a new figure. Otherwise, update the figure.
        Figure consists of three subplots:
            1. Original image
            2. Current image
            3. Corrupted image
        """
            
        observation = self.apply_cumulative_actions( )

        if not fig_in :

            fig = plt.figure()
            ax_orig = fig.add_subplot( 1 , 3 , 1 )
            ax_orig.set_title( "Original" )
            im_orig = ax_orig.imshow( np.squeeze( self._state_img_orig ) )
            ax_current = fig.add_subplot( 1 , 3 , 2 )
            ax_current.set_title( "Observed" )
            im_current = ax_current.imshow( np.squeeze( observation ) )
            ax_corr = fig.add_subplot( 1 , 3 , 3 )
            ax_corr.set_title( "Corrupted" )
            im_corr = ax_corr.imshow( np.squeeze( self._state_img_corrupted ) )
            fig.show()

        else :

            ax_orig = fig_in.axes[ 0 ]
            ax_orig.images[ 0 ].set_data( self._state_img_orig )
            ax_current = fig_in.axes[ 1 ]
            ax_current.images[ 0 ].set_data( observation )
            ax_corr = fig_in.axes[ 2 ]
            ax_corr.images[ 0 ].set_data( self._state_img_corrupted )
            fig_in.canvas.draw()
            fig_in.canvas.flush_events()
            fig = fig_in

        return fig
    

class MnistClassifier( ABC ) :

    def __init__( self ) :
        (self.ds_train, self.ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        self.ds_train = self.ds_train.map( normalize_image )
        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.shuffle( ds_info.splits['train'].num_examples )
        self.ds_train = self.ds_train.batch( 128 )
        self.ds_train = self.ds_train.prefetch( tf.data.AUTOTUNE )

        self.ds_test = self.ds_test.map( normalize_image )
        self.ds_test = self.ds_test.batch( 128 )
        self.ds_test = self.ds_test.cache()
        self.ds_test = self.ds_test.prefetch( tf.data.AUTOTUNE )

    @abstractmethod
    def train( self ) :
        """
        A method which trains the classifier on the MNIST data set.
        """
        pass

    @abstractmethod
    def classify( self , image_in ) :
        """ 
        A method which takes in an image and returns the inferred class.
        Args:
            image_in : a 1xMxM MNIST image as a numpy ndarray M is the image size (square).
        Returns:
            The index of the class that the classifier thinks the image belongs to.
        """ 
        pass

class MnistClassifier_DNN( MnistClassifier ) :

    def __init__( self ) :

        super(MnistClassifier_DNN, self).__init__()

        self.model = tf.keras.models.Sequential( [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ] )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    def train( self ) :
        self.model.fit(
            self.ds_train,
            epochs=6,
            validation_data=self.ds_test,
        )
 
    def classify( self , image_in ) :
 
        class_inferences = np.squeeze( self.model.predict( np.rollaxis( image_in , 2 , 0 ) ) )
        # normalize the inferences
        class_inferences = class_inferences / np.sum( class_inferences )

        return class_inferences