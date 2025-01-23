## NEURAL NETWORK DEFINITION AND DEPENDENCIES

import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
import tensorflow_hub as hub
import tensorflow_text as text
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
 

    
class Supervised_ML_Contrastive_Jaccard_threshold(tf.keras.losses.Loss):
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
                 distance: str = "euclidean",
                 jaccard_threshold:float = 0.2):

        super().__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.distance = distance
        self.jaccard_threshold = jaccard_threshold

    @tf.function
    def call(self, y_true_ml, z):

        batch_size = tf.cast(tf.shape(y_true_ml)[0],tf.float32)
        y_true_ml = tf.expand_dims(y_true_ml, 1) #[bsz,rels]-> [bsz,1,rels]

        ## create weighted beta mask: consider y&yT and normalize each row
        ## [bsz,1,rel] [1,bsz,rel] --> [bsz, bsz, rel] -> [bsz,bsz]
        and_mask = tf.reduce_sum( tf.cast(y_true_ml * tf.transpose(y_true_ml, [1, 0, 2]), tf.float32), -1)
        or_mask = tf.reduce_sum( tf.cast(y_true_ml | tf.transpose(y_true_ml, [1, 0, 2]), tf.float32), -1)
        beta_mask = and_mask/or_mask

        beta_mask = remove_diag(beta_mask)
        beta_mask = tf.where(beta_mask >= self.jaccard_threshold, beta_mask , 0.)
        
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
########### Initialize Bert
##########################################################################

def init_BERT(only_real_token=False, extract_CLS_only=False):
    
    preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    
    # Step 1: tokenize batches of text inputs.
    text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string)] # 2 input layers for 2 text inputs
    tokenize = hub.KerasLayer(preprocessor.tokenize)
    tokenized_inputs = [tokenize(segment) for segment in text_inputs]
    
    # Step 3: pack input sequences for the Transformer encoder.
    seq_length = 512  # Your choice here.
    bert_pack_inputs = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length))  # Optional argument.
    encoder_inputs = bert_pack_inputs(tokenized_inputs)
    
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        trainable=False)
    outputs = encoder(encoder_inputs)
    # Get the output of the [CLS] token, which represents the sentence embedding.
    pooled_output = outputs["pooled_output"]      # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
    
    # Obtain the token mask indicating which tokens belong to the actual input sentences.
    input_mask = encoder_inputs["input_mask"]
    
    # Cast input mask to float32 to match the data type of sequence output.
    input_mask = tf.cast(input_mask, dtype=tf.float32)
    
    # Apply the input mask to filter out padding tokens.
    filtered_sequence_output = sequence_output * tf.expand_dims(input_mask, axis=-1)
    
    if only_real_token:
        # Keep only tokens from the original input sentence (excluding [CLS] and [SEP])
        start_token_index = 1  # Start after [CLS]
        
        # Compute the end token index based on the input_mask
        end_token_index = tf.reduce_sum(input_mask, axis=-1) - 1
        
        # Squeeze the end_token_index tensor to remove extra dimensions
        end_token_index = tf.squeeze(end_token_index, axis=-1)
        
        # Convert end_token_index to a scalar tensor
        end_token_index = tf.cast(end_token_index, dtype=tf.int32)
        
        # Create a range of indices from start_token_index to end_token_index
        indices = tf.range(start_token_index, end_token_index)
        filtered_sequence_output = filtered_sequence_output[:, start_token_index:end_token_index]
    
    if extract_CLS_only:
        embedding_model = tf.keras.Model(text_inputs, pooled_output)  # Extract [CLS] token embedding if you put pooled_output.
    else:
        embedding_model = tf.keras.Model(text_inputs, filtered_sequence_output) # Extract tokens masked embedding if you put filtered_sequence_output.
    
    #sentences = tf.constant([sentence])
    return embedding_model, preprocessor,tokenize


##########################################################################
########### Sentence Attention
##########################################################################

class SentenceAttention(tf.keras.Model):
    """
    token-level attention for passage-level relation extraction.
    """

    def __init__(self, LLM_encoder, attention_features, dense_neurons=3, output_dim=20, activation='relu'):
        """
        Args:
            LLM_encoder: encoder for whole passage (bag of sentences)
            attention_features: number of attention feautures
        """
        super(SentenceAttention, self).__init__()
        self.LLM_encoder = LLM_encoder
        self.embed_dim = self.LLM_encoder.layers[-1].output_shape[-1]
        self.attention_features = attention_features
        self.fc_output = tf.keras.layers.Dense(dense_neurons, activation=activation)
        
        # Define relation_embeddings as a trainable weight
        self.relation_embeddings = self.add_weight(shape=(self.attention_features, self.embed_dim),
                                                   initializer=tf.keras.initializers.GlorotNormal(),
                                                   trainable=True,
                                                   name='relation_embeddings')
                                                   
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.output_dim = tf.keras.layers.Dense(output_dim, activation=activation)
        self.lambda_layer=tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    @tf.function
    def call(self, batch_sentence, train=True):
        """
        Args:
            token: [batch,L], index of tokens   [batch,L]
        Return:
            logits, (B, N)
            Prova Contrastive: R=20, Dense=3
        """
        #batch_size = tf.shape(rep)[0]
        batch_size = tf.shape(batch_sentence)[0]
        #Comment if you want to head tail cls attention and not sentence
        rep = self.LLM_encoder(batch_sentence)  # [batch,L,emb_dim]
        
        
        att_mat = tf.tile(tf.expand_dims(self.relation_embeddings, axis=0), [batch_size, 1, 1])  #  [batch,R,emb_dim]
        att_scores = tf.transpose(tf.matmul(rep, tf.transpose(att_mat, perm=[0, 2, 1])), perm=[0, 2, 1])  #    [batch,R,L]
        att_scores = self.softmax(att_scores) #      [batch,R,L]
        rel_logits = tf.matmul(att_scores, rep) #    [batch,R,L]x[batch,L,emb_dim] -> [batch,R,emb_dim]
        rel_scores = self.fc_output(rel_logits) # [batch,R,emb_dim] -> [batch,R,dense_neurons] -> [batch,R,dense_neurons]
        rel_scores = self.flatten(rel_scores) # [batch,R,dense_neurons] -> [batch,R*dense_neurons]
        rel_scores = self.output_dim(rel_scores)  # [batch,R*dense_neurons] -> [batch,output_dim]
        rel_scores = self.lambda_layer(rel_scores)

        

        return rel_scores



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
    

    
    if architecture not in ["MLP","CNN","PARE_CL"]:
        raise KeyError(f"Allowed architectures are CNN or MLP, got {architecture} instead")
   
    if architecture=='MLP':
        input_shape=training_configuration['input_shape']
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        model=MLP(input_shape,depth,pert,output_dimensions,activation=activation)
    
    elif architecture=='PARE_CL':
        LLM_encoder, _, _ = init_BERT()
        #tf.keras.backend.clear_session()
        
        model=SentenceAttention(LLM_encoder,
            output_dim=training_configuration['output_dim'],
            attention_features=training_configuration['attention_features'],
            dense_neurons=training_configuration['dense_neurons'],
            activation=training_configuration['activation'])

        

    model.compile() 
    model.build(input_shape=(None, 1))

    print('model weights skeleton', sum([el.size for el in model.get_weights()]))
    
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
    
    
    mlcl_loss=training_configuration['loss']
    
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
        
    if mlcl_loss=='similarity':
        loss = Supervised_ML_Contrastive(
            temperature=temperature,
            base_temperature=base_temperature,
            distance=distance,
        )
    elif mlcl_loss=='jaccard':
        jaccard_threshold = training_configuration['jaccard_threshold']
        loss = Supervised_ML_Contrastive_Jaccard_threshold(
            temperature=temperature,
            base_temperature=base_temperature,
            distance=distance,
            jaccard_threshold=jaccard_threshold
        )

    model.compile(
        loss=loss, optimizer=optimizer, run_eagerly=False
    )  
    
    return model


