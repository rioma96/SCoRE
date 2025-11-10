## NEURAL NETWORK DEFINITION AND DEPENDENCIES

import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Dropout
import tensorflow_hub as hub
import tensorflow_text as text
import os, math
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

###########################################################################
########### HIERARCHICAL SUPERVISED CONTRASTIVE LOSS
############################################################################


class Hierarchical_Supervised_ML_Contrastive(tf.keras.losses.Loss):
    """
    Supervised multi-label hierarchical contrastive loss:
    - Fine-grained loss on y_true_fine_ml (e.g., Person/President/Obama)
    - Coarse-grained loss on y_true_coarse_ml (e.g., Person/President)

    Args:
        temperature : scaling of similarities
        base_temperature : normalization constant
        distance : euclidean or cosine
        lambda_coarse : weight for the coarse-level loss
    """

    def __init__(self,
                 temperature: float = 0.5,
                 base_temperature: float = 1.0,
                 distance: str = "euclidean",
                 lambda_coarse: float = 0.5,
                 number_of_classes: int = 0,
                 coarse_labels_temperature: float = 0.5):  # default: equally weight coarse and fine

        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.distance = distance
        self.lambda_coarse = lambda_coarse
        self.number_of_classes = number_of_classes
        self.coarse_labels_temperature = coarse_labels_temperature

    def compute_contrastive_loss(self, y_true_ml, z):
        batch_size = tf.cast(tf.shape(y_true_ml)[0], tf.float32)
        y_true_ml = tf.expand_dims(y_true_ml, 1)

        beta_mask = tf.reduce_sum(
            tf.cast(y_true_ml * tf.transpose(y_true_ml, [1, 0, 2]), tf.float32), -1
        )
        beta_mask = remove_diag(beta_mask)
        beta_mask /= tf.maximum(tf.reduce_sum(beta_mask, -1, keepdims=True), 1)

        distances = tf.cast(distance_matrix(z, self.distance), tf.float32)
        distances = remove_diag(distances)

        logits = distances / self.temperature
        exp_logits = tf.exp(-logits)

        loss = beta_mask * (
            logits + tf.math.log(1e-10 + tf.reduce_sum(exp_logits, axis=1, keepdims=True))
        )

        return (self.temperature / self.base_temperature) * tf.reduce_sum(loss) / batch_size

    def compute_contrastive_loss_coarse(self, y_true_ml, z):
        batch_size = tf.cast(tf.shape(y_true_ml)[0], tf.float32)
        y_true_ml = tf.expand_dims(y_true_ml, 1)

        beta_mask = tf.reduce_sum(
            tf.cast(y_true_ml * tf.transpose(y_true_ml, [1, 0, 2]), tf.float32), -1
        )
        beta_mask = remove_diag(beta_mask)
        beta_mask /= tf.maximum(tf.reduce_sum(beta_mask, -1, keepdims=True), 1)

        distances = tf.cast(distance_matrix(z, self.distance), tf.float32)
        distances = remove_diag(distances)

        logits = distances / self.coarse_labels_temperature
        exp_logits = tf.exp(-logits)

        loss = beta_mask * (
            logits + tf.math.log(1e-10 + tf.reduce_sum(exp_logits, axis=1, keepdims=True))
        )

        return (self.coarse_labels_temperature / self.base_temperature) * tf.reduce_sum(loss) / batch_size


    @tf.function
    def call(self, y_true_ml, z):
        """
        Args:
            inputs: tuple of (y_true_fine_ml, y_true_coarse_ml, z)
        Returns:
            Combined contrastive loss
        """

        # y_true_fine_ml, y_true_coarse_ml = y_true_ml

        y_true_fine_ml = y_true_ml[:, :self.number_of_classes]
        y_true_coarse_ml = y_true_ml[:, self.number_of_classes:]

        loss_fine = self.compute_contrastive_loss(y_true_fine_ml, z)
        loss_coarse = self.compute_contrastive_loss(y_true_coarse_ml, z)

        return (1.0 - self.lambda_coarse) * loss_fine + self.lambda_coarse * loss_coarse

##########################################################################


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


class Supervised_ML_Contrastive_MultiMLPs(tf.keras.losses.Loss):
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
                 add_NA_class: bool = False
                 #thresholds = None
                 ):

        super().__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.distance = distance
        self.add_NA_class = add_NA_class
        #self.thresholds = thresholds

    @tf.function
    def call(self, y_true_ml, z):

        # multiplicative_factor=tf.expand_dims(tf.math.sqrt(tf.cast(self.thresholds,tf.float32)),0)
        # multiplicative_factor=tf.expand_dims(tf.cast(self.thresholds,tf.float32),0)
        # batch_temperature_ml=multiplicative_factor*tf.cast(y_true_ml,tf.float32)
        # batch_temperature_ml_regularized= tf.cast(batch_temperature_ml<1e-20,tf.float32)*1e5 + batch_temperature_ml
        # batch_temperature=tf.reduce_min(batch_temperature_ml_regularized,axis=1, keepdims=True) * self.temperature

        batch_size = tf.cast(tf.shape(y_true_ml)[0],tf.float32)
        if self.add_NA_class:
            print(y_true_ml)
            existing_classes=tf.cast(tf.reduce_sum(y_true_ml,axis=1,keepdims=True)>0, tf.float32)
            negative_class=tf.cast(tf.ones((tf.shape(y_true_ml)[0],1)),tf.float32)-existing_classes
            y_true_ml = tf.concat([y_true_ml,negative_class],axis=1)



        y_true_ml = tf.expand_dims(y_true_ml, 1)


        ## create weighted beta mask: consider y&yT and normalize each row
        ## [bsz,1,10] [1,bsz,10] --> [bsz, bsz, 10] -> [bsz,bsz]
        beta_mask = tf.reduce_sum(
            tf.cast(y_true_ml * tf.transpose(y_true_ml, [1, 0, 2]), tf.float32), -1
        )
        beta_mask = remove_diag(beta_mask)
        beta_mask /= tf.maximum(tf.reduce_sum(beta_mask, -1, keepdims=True), 1)
        
        ## create boolean beta mask
        # bool_mask = tf.cast(beta_mask > 1e-7, tf.float32)

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

###############################
###### Temperature adapter function
###############################

class TemperatureSchedulerAuto(tf.keras.callbacks.Callback):
    """
    Monotonic non-increasing temperature schedule with a hard floor at T_end.
    Once T_end is reached, temperature never increases again.
    If freeze_at_end=True, we stop updating after hitting T_end.
    """
    def __init__(self, T_start: float, T_end: float,
                 mode: str = "cosine",        # "cosine" | "exp" | "linear"
                 per_batch: bool = True,
                 warmup_epochs: int = 0,
                 hold_epochs: int = 0,
                 freeze_at_end: bool = True):
        super().__init__()
        self.T_start = float(T_start)
        self.T_end = float(T_end)
        self.mode = mode
        self.per_batch = per_batch
        self.warmup_epochs = int(warmup_epochs)
        self.hold_epochs = int(hold_epochs)
        self.freeze_at_end = bool(freeze_at_end)

        self._global_step = 0
        self._steps_per_epoch = None
        self._max_epochs = None
        self._temp_vars = []
        self._frozen = False

    @staticmethod
    def _set_all(temp_vars, value: float):
        for v in temp_vars:
            v.assign(value)

    def _interp(self, p: float) -> float:
        # p in [0,1]; compute target WITHOUT clamping to T_end yet
        p = min(max(p, 0.0), 1.0)
        if self.mode == "cosine":
            return self.T_end + 0.5 * (self.T_start - self.T_end) * (1 + math.cos(math.pi * p))
        elif self.mode == "exp":
            return self.T_start * (self.T_end / self.T_start) ** p
        else:  # linear
            return self.T_start + (self.T_end - self.T_start) * p

    def _target_T(self, epoch_float: float) -> float:
        # Warmup -> hold -> decay; progress clamped to [0,1]
        if epoch_float < self.warmup_epochs:
            T = self.T_start
        else:
            epoch_float -= self.warmup_epochs
            if epoch_float < self.hold_epochs:
                T = self.T_start
            else:
                epoch_float -= self.hold_epochs
                denom = max(1.0, (self._max_epochs - self.warmup_epochs - self.hold_epochs - 1))
                progress = epoch_float / denom
                T = self._interp(progress)
        # Hard floor: never go below T_end (monotone non-increasing to >= T_end)
        return max(self.T_end, T)

    def _apply(self, epoch_float: float):
        if not self._temp_vars or self._frozen:
            return
        T = self._target_T(epoch_float)
        self._set_all(self._temp_vars, T)
        # Optional: once at floor, stop updating for good
        if self.freeze_at_end and T <= self.T_end + 1e-12:
            self._frozen = True

    # ---- Keras hooks ----
    def on_train_begin(self, logs=None):
        self._max_epochs = int(self.params.get('epochs', 1))
        self._steps_per_epoch = self.params.get('steps')  # may be None for finite datasets
        self._temp_vars = getattr(self.model, "_temperature_vars", None) or []
        if not self._temp_vars:
            tf.print("[TemperatureSchedulerAuto] No model._temperature_vars found; scheduler is a no-op.")
        if self.per_batch and (self._steps_per_epoch is None):
            tf.print("[TemperatureSchedulerAuto] steps_per_epoch unknown; falling back to per-epoch updates.")
            self.per_batch = False
        # Initialize
        self._apply(0.0)

    def on_epoch_begin(self, epoch, logs=None):
        if not self.per_batch:
            self._apply(float(epoch))

    def on_train_batch_begin(self, batch, logs=None):
        if self.per_batch and self._steps_per_epoch:
            epoch_idx = self._global_step / float(self._steps_per_epoch)
            self._apply(epoch_idx)
            self._global_step += 1

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and self._temp_vars:
            logs['temperature'] = float(self._temp_vars[0].numpy())

#########################################################################
########## Loss Adaptive
#########################################################################
class Supervised_ML_Contrastive_Adaptive(tf.keras.losses.Loss):
    """
    Supervised multi-label contrastive loss
    """
    def __init__(self,
                 temperature: float = 0.5,
                 base_temperature: float = 1.0,
                 distance: str = "euclidean"):
        super().__init__()
        # <-- key change: make it mutable so a callback can anneal it
        self.temperature = tf.Variable(float(temperature), dtype=tf.float32, trainable=False, name="temperature")
        self.base_temperature = tf.constant(float(base_temperature), dtype=tf.float32)
        self.distance = distance

    @tf.function
    def call(self, y_true_ml, z):
        batch_size = tf.cast(tf.shape(y_true_ml)[0], tf.float32)
        y_true_ml = tf.expand_dims(y_true_ml, 1)

        beta_mask = tf.reduce_sum(
            tf.cast(y_true_ml * tf.transpose(y_true_ml, [1, 0, 2]), tf.float32), -1
        )
        beta_mask = remove_diag(beta_mask)
        beta_mask /= tf.maximum(tf.reduce_sum(beta_mask, -1, keepdims=True), 1)

        distances = tf.cast(distance_matrix(z, self.distance), tf.float32)
        distances = remove_diag(distances)

        logits = distances / self.temperature

        exp_logits = tf.exp(-logits)
        exp_logits = remove_diag(exp_logits)

        loss = beta_mask * (
            logits + tf.math.log(1e-10 + tf.reduce_sum(exp_logits, axis=1, keepdims=True))
        )
        loss = (self.temperature / self.base_temperature) * tf.reduce_sum(loss) / batch_size
        return loss

    
##########################################################################
########### Model architectures
##########################################################################


def MLP(input_shape,depth,pert,output_dimensions,activation='relu'): 

    neurons_per_layer = round(depth/pert)
    layers=[
            Input(shape=(input_shape,)),
            Flatten(),
            *[Dense(neurons_per_layer, activation=activation) for i in range(depth)]]
    
    if output_dimensions:
        layers.append(Dense(output_dimensions, activation=activation))
        
    layers.append(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model=tf.keras.models.Sequential(layers)
        
    return model

def MLP_DROPOUT(input_shape,depth,pert,output_dimensions,activation='relu', dropout_rate=0.3): 
    neurons_per_layer = round(depth/pert)
    layers = [
        Input(shape=(input_shape,)),
        Flatten()
    ]
    # Blocchi Dense + Dropout
    for i in range(depth):
        layers.append(Dense(neurons_per_layer, activation=activation))
        if i == 1:
            layers.append(Dropout(dropout_rate))
    if output_dimensions:
        layers.append(Dense(output_dimensions, activation=activation))
    layers.append(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model = tf.keras.models.Sequential(layers)
    return model


def multiple_MLP_encoding(input_shape,
                        depth,
                        pert, 
                        output_dimensions, 
                        n_branches,
                        activation='relu'):
    width = round(depth/pert)
    def create_mlp_branch(inputs, depth, width, output_dimensions, activation, branch_index):
        x = Flatten()(inputs)
        for i in range(depth):
            x = Dense(width, activation=activation)(x)
        
        if output_dimensions:
            x = Dense(output_dimensions, activation=activation)(x)
        
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name=f'branch_{branch_index}_norm')(x)
        return x

    # Create input layer
    input_l = Input(shape=(input_shape,))

    # Create multiple MLP branches
    outputs = []
    for i in range(n_branches):
        outputs.append(create_mlp_branch(input_l, depth, width, output_dimensions, activation, i))

    # Create the model with one input and multiple outputs
    model = tf.keras.models.Model(inputs=input_l, outputs=outputs)
    return model 




def multiple_MLP_encoding_common_projection(input_shape,
                                          n_branches,
                                          depth,
                                          depth_encoder,
                                          pert, 
                                          output_dimensions, 
                                          activation='relu'):
    # Create input layer
    input_l = Input(shape=(input_shape,))
    width = round(depth/pert)
    # Create shared encoder (as a reusable layer)
    def create_shared_encoder():
        shared = tf.keras.Sequential()
        for i in range(depth_encoder):  # Using depth_encoder parameter
            shared.add(Dense(output_dimensions, activation=activation))
        return shared
    
    # Create the shared encoder once to reuse
    shared_encoder = create_shared_encoder()
    
    # Function to create an MLP branch
    def create_simple_mlp_branch(inputs):
        x = Flatten()(inputs)
        for i in range(depth):
            x = Dense(width, activation=activation)(x)
        return x
    
    # Create multiple MLP branches and pass their outputs through the shared encoder
    outputs = []
    for i in range(n_branches):
        branch_output = create_simple_mlp_branch(input_l)
        encoded_output = shared_encoder(branch_output)
        normalized_output = Lambda(lambda x: tf.math.l2_normalize(x, axis=1),  
                                 name=f'branch_{i}_norm')(encoded_output)
        outputs.append(normalized_output)
    
    # Create the model with one input and multiple outputs
    model = tf.keras.models.Model(inputs=input_l, outputs=outputs)
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

def CreateModelSkeleton(training_configuration:dict, branch_idx=None):
    
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
    

    
    if architecture not in ["MLP","CNN","PARE_CL","COMAL_NO_SHARED","COMAL_SHARED",'MLP_DROPOUT']:
        raise KeyError(f"Allowed architectures are CNN or MLP, got {architecture} instead")
   
    if architecture=='MLP':
        input_shape=training_configuration['input_shape']
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        model=MLP(input_shape,depth,pert,output_dimensions,activation=activation)
        model.compile() 

    elif architecture=='MLP_DROPOUT':
        input_shape=training_configuration['input_shape']
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        dropout_rate=training_configuration['dropout_rate']
        model=MLP_DROPOUT(input_shape,depth,pert,output_dimensions,activation=activation, dropout_rate=dropout_rate)
        model.compile()
    
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

    elif architecture=='COMAL_NO_SHARED':
        input_shape=training_configuration['input_shape']
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        nbranches=training_configuration['n_branches']
        model=multiple_MLP_encoding(input_shape,depth,pert,output_dimensions,nbranches,activation=activation)
        if branch_idx is not None:
        # Ritorna solo il sotto-modello per il branch richiesto
            return tf.keras.Model(inputs=model.input, outputs=model.outputs[branch_idx])
        model.compile() 

    elif architecture=='COMAL_SHARED':
        input_shape=training_configuration['input_shape']
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        depth_encoder=training_configuration['depth_encoder']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        nbranches=training_configuration['n_branches']
        model=multiple_MLP_encoding_common_projection(input_shape,n_branches=nbranches,depth=depth,depth_encoder=depth_encoder,pert=pert,output_dimensions=output_dimensions,activation=activation)
        if branch_idx is not None:
            # Ritorna solo il sotto-modello per il branch richiesto
            return tf.keras.Model(inputs=model.input, outputs=model.outputs[branch_idx])
        model.compile()


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
    
    if architecture not in ["MLP","CNN","COMAL_NO_SHARED","COMAL_SHARED",'MLP_DROPOUT']:
        raise KeyError(f"Allowed architectures are CNN or MLP, got {architecture} instead")
    
    if distance not in ["cosine","euclidean","euclidean_squared","cosine_triangle"]:
        raise KeyError(f"Allowed distances are [cosine, euclidean, euclidean_squared, cosine_triangle], got {distance} instead")
        
    
    if architecture=='MLP':
        
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        model=MLP(input_shape,depth,pert,output_dimensions,activation=activation)

    elif architecture=='MLP_DROPOUT':
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        dropout_rate=training_configuration['dropout_rate']
        model=MLP_DROPOUT(input_shape,depth,pert,output_dimensions,activation=activation, dropout_rate=dropout_rate)
        
    elif architecture=='CNN':
        
        activation=training_configuration['activation']
        filter_unit=training_configuration['filter_unit']
        output_dimensions=training_configuration['output_dimensions']
        model=CNN(input_shape,filter_unit,output_dimensions,activation=activation)

    elif architecture=='COMAL_NO_SHARED':
        input_shape=training_configuration['input_shape']
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        nbranches=training_configuration['n_branches']
        model=multiple_MLP_encoding(input_shape,depth,pert,output_dimensions,nbranches,activation=activation)

    elif architecture=='COMAL_SHARED':
        input_shape=training_configuration['input_shape']
        activation=training_configuration['activation']
        depth=training_configuration['depth']
        depth_encoder=training_configuration['depth_encoder']
        pert=training_configuration['pert']
        output_dimensions=training_configuration['output_dimensions']
        nbranches=training_configuration['n_branches']
        model=multiple_MLP_encoding_common_projection(input_shape,n_branches=nbranches,depth=depth,depth_encoder=depth_encoder,pert=pert,output_dimensions=output_dimensions,activation=activation)
        
    if mlcl_loss=='similarity':
        loss = Supervised_ML_Contrastive(
            temperature=temperature,
            base_temperature=base_temperature,
            distance=distance,
        )
    elif mlcl_loss=='similarity_adaptive':
        loss = Supervised_ML_Contrastive_Adaptive(
            temperature=temperature,
            base_temperature=base_temperature,
            distance=distance,
        )
        temperature_vars = [loss.temperature]

    elif mlcl_loss=='hierarchical_similarity':
        lambda_coarse = training_configuration.get('lambda_coarse', 0.5)
        loss = Hierarchical_Supervised_ML_Contrastive(
            temperature=temperature,
            base_temperature=base_temperature,
            distance=distance,
            lambda_coarse=lambda_coarse,
            number_of_classes=training_configuration['number_of_classes'],
            coarse_labels_temperature=training_configuration['coarse_labels_temperature']
        )
    elif mlcl_loss=='jaccard':
        jaccard_threshold = training_configuration['jaccard_threshold']
        loss = Supervised_ML_Contrastive_Jaccard_threshold(
            temperature=temperature,
            base_temperature=base_temperature,
            distance=distance,
            jaccard_threshold=jaccard_threshold
        )
    elif mlcl_loss=='similarity_multiMLP':
        loss_dict = {}
        add_NA_class = training_configuration['add_NA_class']
        nbranches=training_configuration['n_branches']
        for i in range(nbranches):
            loss_instance = Supervised_ML_Contrastive_MultiMLPs(
                temperature=temperature,
                base_temperature=base_temperature,
                distance=distance,
                add_NA_class=add_NA_class
            )
            loss_dict[f"branch_{i}_norm"] = loss_instance

    if mlcl_loss=='similarity_multiMLP':
        model.compile(
            loss=loss_dict, optimizer=optimizer, run_eagerly=False
        )
    else:
        model.compile(
            loss=loss, optimizer=optimizer, run_eagerly=False
        )  
    if mlcl_loss=='similarity_adaptive':
        model._temperature_vars = temperature_vars 
    
    return model


