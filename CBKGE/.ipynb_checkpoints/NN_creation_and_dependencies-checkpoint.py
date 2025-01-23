## NEURAL NETWORK DEFINITION AND DEPENDENCIES

import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


##########################################################################
########### DISTANCES FOR HIDDEN REPRESENTATION SPACE
##########################################################################

@tf.function
def distance_matrix(z, dist="euclidean"):
    """
    Compute distance matrix in the embedding space
    z = embeddings of size [ batch , embedding_dims ]
    dist = type of distance definition to be used.
    Possible chices are:
        - 'euclidean_squared'
        - 'cosine'
        - 'euclidean'
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(z, tf.transpose(z))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)
        
    # Compute the pairwise distance matrix as:
    if dist == "euclidean_squared":
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = (
            tf.expand_dims(square_norm, 1)
            - 2.0 * dot_product
            + tf.expand_dims(square_norm, 0)
        )
        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)
    elif dist == "cosine_triangle":
        # sqrt( 2 + eps - 2 (a . b)/|a||b| )
        # shape (batch_size, batch_size)
        distances = tf.math.sqrt(
            2.0
            + 1e-10
            - 2.0
            * dot_product
            / tf.math.sqrt(
                tf.expand_dims(square_norm, 1) * tf.expand_dims(square_norm, 0) + 1e-10
            )
        )
    elif dist == "cosine":
        # sqrt( 2 + eps - 2 (a . b)/|a||b| )
        # shape (batch_size, batch_size)
        distances = (
            1.0
            + 1e-10
            - 1.0
            * dot_product
            / tf.math.sqrt(
                tf.expand_dims(square_norm, 1) * tf.expand_dims(square_norm, 0) + 1e-10
            )
        )
    elif dist == "euclidean":
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = (
            tf.expand_dims(square_norm, 1)
            - 2.0 * dot_product
            + tf.expand_dims(square_norm, 0)
        )
        distances = tf.maximum(distances, 0.0)
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-20
        distances = tf.sqrt(distances)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances




#########################
##### MULTILABEL CONTRASTIVE LOSS
#################################

class Supervised_ML_Contrastive(tf.keras.losses.Loss):
    """
    Supervised multi-label contrastive loss to use in distant-to-fully supervised training
    Args:
        temperature : to increase or decrease magnitude of attraction/repulsion
        base_temperature : to rescale the loss value
        distance : type of metric to be used in evaluating distances

        y_true_ml: true multi-labels of the batch of shape [bsz,n_classes]
        z: hidden vector of shape [bsz, n_features].
    """

    def __init__(self, 
                 temperature:float = 0.5, 
                 base_temperature:float = 1.0, 
                 distance: str = "euclidean"):

        super().__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.distance = distance

    @tf.function
    def call(self, y_true_ml, z):

        batch_size = tf.cast(tf.shape(y_true_ml)[0],tf.float32)
        y_true_ml = tf.expand_dims(y_true_ml, 1)

        ## create weighted beta mask: consider y&yT and normalize each row
        ## [bsz,1,10] [1,bsz,10] --> [bsz, bsz, 10] -> [bsz,bsz]
        beta_mask = tf.reduce_sum(
            tf.cast(y_true_ml * tf.transpose(y_true_ml, [1, 0, 2]), tf.float32), -1
        )
        beta_mask = remove_diag(beta_mask)
        beta_mask /= tf.maximum(tf.reduce_sum(beta_mask, -1, keepdims=True), 1)
        
        ## create boolean beta mask
        bool_mask = tf.cast(beta_mask > 1e-7, tf.float32)

        ### Distance matrix and logits
        distances = tf.cast(distance_matrix(z, self.distance), tf.float32)
        distances = remove_diag(distances)

        logits = tf.divide(distances, self.temperature)

        # compute mean of log-likelihood over positive
        exp_logits = tf.exp(-logits)
        exp_logits = remove_diag(exp_logits)

        loss = beta_mask * (
            logits
            + tf.math.log(1e-10 + tf.reduce_sum(exp_logits, axis=1, keepdims=True))
        )

        loss = (
            (self.temperature / self.base_temperature)
            * tf.reduce_sum(loss)
            / batch_size
        )
        
        return loss
    
    
@tf.function
def remove_diag(tensor):
    # set tensor diagonal elements to zero
    tensor = tensor - tf.linalg.tensor_diag(tf.linalg.diag_part(tensor))
    return tensor

    
##########################################################################
########### Model architectures
##########################################################################


def MLP(input_shape,depth,pert,output_dimensions,activation='relu'): 

    neurons_per_layer = round(depth/pert)
    layers=[
            Input(shape=(input_shape)),
            Flatten(),
            *[Dense(neurons_per_layer, activation=activation) for i in range(depth)]]
    
    if output_dimensions:
        layers.append(Dense(output_dimensions, activation=activation))
        
    layers.append(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model=tf.keras.models.Sequential(layers)
        
    return model


    
##########################################################################
########### Compute NN accuracy on dataset given a margin
##########################################################################

def CreateModelSkeleton(training_configuration:dict):
    
    '''
    
    Create and compile model for an input tensor having shape input_shape (tuple),
    using specifics included in training_configuration. 
    The following key are required in training_configuration:
    
        - 'input_shape': tuple
        - 'architecture' : str in ['MLP','CNN']
        - 'activation': str or tf.keras.activations 
        - 'neurons_per_layer': int   (for MLP)
        - 'pert': float              (for MLP)
        - 'filter_unit':int          (for CNN)
        - 'output_dimensions':int    (for CNN/MLP)
    
    '''
    
    architecture=training_configuration['architecture']
    input_shape=training_configuration['input_shape']

    
    if architecture not in ["MLP","CNN"]:
        raise KeyError(f"Allowed architectures are CNN or MLP, got {architecture} instead")
   
    if architecture=='MLP':
        
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        model=MLP(input_shape,depth,pert,output_dimensions,activation=activation)
        

    model.compile()  
    
    return model


def CreateModel(training_configuration:dict):
    
    '''
    
    Create and compile model for an input tensor having shape input_shape (tuple),
    using specifics included in training_configuration. 
    The following key are required in training_configuration:
    
        - 'input_shape': tuple
        - 'architecture' : str in ['MLP','CNN']
        - 'learning_rate' : float
        - 'optimizer' : tf.keras.optimizers
        - 'distance' : str in ['cosine', 'euclidean', 'euclidean_squared', 'cosine_triangle']
        - 'base_temperature': float
        - 'temperature': float
        - 'activation': str or tf.keras.activations 
        - 'neurons_per_layer': int   (for MLP)
        - 'pert': float              (for MLP)
        - 'filter_unit':int          (for CNN)
        - 'output_dimensions':int    (for CNN/MLP)
    
    '''
    
    architecture=training_configuration['architecture']
    input_shape=training_configuration['input_shape']
    
    learning_rate=training_configuration['learning_rate']
    optimizer=tf.keras.optimizers.deserialize(training_configuration['optimizer'])
    optimizer.learning_rate.assign(learning_rate)

    distance=training_configuration['distance']
    base_temperature=training_configuration['base_temperature']
    temperature=training_configuration['temperature']
    
    if architecture not in ["MLP","CNN"]:
        raise KeyError(f"Allowed architectures are CNN or MLP, got {architecture} instead")
    
    if distance not in ["cosine","euclidean","euclidean_squared","cosine_triangle"]:
        raise KeyError(f"Allowed distances are [cosine, euclidean, euclidean_squared, cosine_triangle], got {distance} instead")
        
    
    if architecture=='MLP':
        
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        model=MLP(input_shape,depth,pert,output_dimensions,activation=activation)
        
    elif architecture=='CNN':
        
        activation=training_configuration['activation']
        filter_unit=training_configuration['filter_unit']
        output_dimensions=training_configuration['output_dimensions']
        model=CNN(input_shape,filter_unit,output_dimensions,activation=activation)
        
    loss = Supervised_ML_Contrastive(
        temperature=temperature,
        base_temperature=base_temperature,
        distance=distance,
    )

    model.compile(
        loss=loss, optimizer=optimizer, run_eagerly=False
    )  
    
    return model


