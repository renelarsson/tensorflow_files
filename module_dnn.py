# Import dependencies
import numpy as np
import pickle
import copy

############################## Dense Layer Class ##############################

# Main layer class
class DenseLayer:

  # Initialize dense layer
  def __init__(self, n_inputs, n_neurons, l1_w_reg=0, l2_w_reg=0, l1_b_reg=0, l2_b_reg=0, norm=0.01):

    # Initialize weights and biases
    self.weights = norm * np.random.randn(n_inputs, n_neurons) # Normalize against exploding gradient (NB: 0.1 weight initialization works better for regression)
    self.biases = np.zeros((1, n_neurons)) # Initialize as as 0 row vectors (may cause vanishing gradient)

    # Set regularizer
    self.l1_w_reg = l1_w_reg
    self.l2_w_reg = l2_w_reg
    self.l1_b_reg = l1_b_reg
    self.l2_b_reg = l2_b_reg

  # Dense layer forward pass method
  def forward(self, inputs, training):

    # Store input values for backpropagation
    self.inputs = inputs
    # Calculate output values
    self.output = np.dot(inputs, self.weights) + self.biases # Weights are initialized with transposed dimensions above

  # Dense layer backward pass
  def backward(self, d_values):

    # Chain rule calculates gradients of the loss w.r.t. the weights
    self.d_weights = np.dot(self.inputs.T, d_values)
    # Sum biases across samples
    self.d_biases = np.sum(d_values, axis=0, keepdims=True)# Keep the gradient as a row vector (0 axis for 1D array) 

    # Regularize weights gradients (L1/L2)
    if self.l1_w_reg > 0:
      # Create matrix of ones with shape of weights
      d_l1 = np.ones_like(self.weights)
      # Add the sign of the weights (L1 partial derivative is either 1 or -1)
      d_l1[self.weights < 0] = -1
      # Multiply L1 by sign gradient matrix and add to d_weights penalizing large gradients (overfitting) 
      self.d_weights += self.l1_w_reg * d_l1

    if self.l2_w_reg > 0:
      # Multiply L2 by weights gradient and add to d_weights penalizing large gradients (overfitting)
      self.d_weights += 2 * self.l2_w_reg * self.weights # L2 partial derivative is 2w

    # Regularize bias gradients (L1/L2)
    if self.l1_b_reg > 0:
      d_l1 = np.ones_like(self.biases)
      d_l1[self.biases < 0] = -1
      self.d_biases += self.l1_b_reg * d_l1

    if self.l2_b_reg > 0:
      self.d_biases += 2 * self.l2_b_reg * self.biases

    # Chain rule calculates gradients w.r.t. inputs from forward method
    self.d_inputs = np.dot(d_values, self.weights.T)

  # Get layer parameters (weights and biases)
  def get_parameters(self):
    return self.weights, self.biases

  # Set layer parameters (weights and biases)
  def set_parameters(self, weights, biases):
    self.weights = weights
    self.biases = biases


############################## Dropout Layer Class ##############################

# Disable neurons to prevent overfitting
class DropoutLayer:

  # Initialize dropout layer
  def __init__(self, rate):
    # Store inverted rate (a rate of 0.2 enables 0.8 neurons)
    self.rate = 1 - rate

  # Forward pass
  def forward(self, inputs, training):

    # Save input values
    self.inputs = inputs
    # Turn off dropout in inference mode (set output to input)
    if not training:
      self.output = inputs.copy()
      return
    # Generate scaled mask - randomly set some values to zero and scale up to keep non-dropout output magnitude
    self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
    # Apply mask to output values
    self.output = inputs * self.binary_mask

  # Backward pass though the same binary mask
  def backward(self, d_values):
    # Apply mask to input gradients
    self.d_inputs = d_values * self.binary_mask


############################## Input Layer Class ##############################

# Non-hidden layer (not weights/biases - called in class Model's finalize method before looping though layers)
class InputLayer:
  # Forward method sets training samples as self.output
  def forward(self, inputs, training):
    self.output = inputs


############################## ReLU Activation Class ##############################

# Non-linear activation/deactivation function
class ReluActivation:

# NB: NO PREDICTION USED - RelU is used to introduce non-linearity for hidden layers, not for output layers.

  def forward(self, inputs, training):

    # Store input values for backpropagation
    self.inputs = inputs
    # Calculate output values from inputs
    self.output = np.maximum(0, inputs) # Negative values are clipped (modified to be zero)

  def backward(self, d_values):

    # Copy the gradients of the loss function w.r.t. the inputs of the ReLU activation layer
    self.d_inputs = d_values.copy()
    # Identify ReLU deactivated inputs (insert zero gradient where input values are negative)
    self.d_inputs[self.inputs <= 0] = 0


############################## Softmax Activation Class ##############################

# Output activation to calculate class probabilities
class SoftmaxActivation:

# NB: NO BACKPROPAGATION USED - see combined Softmax Activation/Categorical Cross-entropy loss in Softmax classifier Class.

  # Use non-negative values and non-linearity (Exponential growth preserves the significance of negative values)
  def forward(self, inputs, training):

    # Store input values (not used but may be relevant for debugging or visualization)
    self.inputs = inputs
    # Keep range between 0 and 1, tackling risk of overflow (as input values approache negative infinity)
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # axis=1 collapses columns, returning row sums
    # Normalize exponentiated values (add up to 1) and keep the output as a 2D array
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)# keepdims returns a column vector (maintaining batch outputs)
    self.output = probabilities

  # Calculate predictions for Softmax outputs (used by class Accuracy for accuracy calculations)
  def predictions(self, outputs):
    # Get highest probability along first axis - see Inspect Binary Cross-entropy
    return np.argmax(outputs, axis=1)
  

############################## Sigmoid Activation Class ##############################

# Output activation to calculate binary probability
class SigmoidActivation:

  # Use non-negative values and non-linearity (Logistic growth preserves the significance of negative values)
  def forward(self, inputs):

    # Store input values for backpropagation
    self.inputs = inputs
    # Calculate Sigmoid
    self.output = 1 / (1 + np.exp(-inputs))

  # Backward pass
  def backward(self, d_values):
    # Calculate derivative from the forward output pass of the sigmoid function
    self.d_inputs = d_values * (1 - self.output) * self.output

  # Calculate predictions for Sigmoid outputs (used by class Accuracy for accuracy calculations)
  def predictions(self, outputs):
    # Return a binary mask of True/False values as an array of 1s/0s (multiply by 1)
    return (outputs > 0.5) * 1
  
############################## Linear Activation Class ##############################

# No activation applied (used for architectural completeness)
class LinearActivation:

  # Forward pass 
  def forward(self, inputs):

    # Store same value for input and output 
    self.inputs = inputs
    self.output = inputs

  # Backward pass
  def backward(self, d_values):
    # Copy the gradients from the next layer, since the derivative of a linear function is 1
    self.d_inputs = d_values.copy()

  # The output values are the predictions themselves
  def predictions(self, outputs):
    return outputs
  

############################## Adam Optimizer Class ##############################

# Adaptive Momentum - uses momentum (as in SGD) and per-weight adaptive learning rate with cache (as in RMSProp)
class AdamOptimizer:

  # Initialize optimizer with settings
  def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):

    # Decrease loss by applying a fraction (LR) to gradient descent path (to be subtracted from weights/biases)
    self.learning_rate = learning_rate
    # Lower learning rate during training
    self.decay = decay
    # Prevent division by 0
    self.epsilon = epsilon
    # Fraction of momentum to apply for bias correction mechanism (momentum and cache zero initialization bias)
    self.beta_1 = beta_1
    # beta_1 defaults to 0.9 -> 0.1, beta_2 defaults to 0.999 -> 0.001
    self.beta_2 = beta_2
    # Decay LR per step (step count fraction)
    self.iterations = 0
    # Adjusted LR during training based on decay and iterations (steps)
    self.current_learning_rate = learning_rate

  # Call once before parameter updates (Prepares for parameter updates)
  def pre_update_params(self):
    # Calculate the adjusted learning rate for the current iteration if there is a decay factor 
    if self.decay:
      # Exponential LR decaying (1/t) - the further in training, the lower the LR value (given larger iterator)
      self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

  # Update parameters (Updates weights and biases)
  def update_params(self, layer):

    # Create zero arrays of parameter shape if layer does not have a cache
    if not hasattr(layer, 'weight_cache'): # Checks if the layer object has attribute named weight_cache
      layer.weight_momentums = np.zeros_like(layer.weights) # Store momentum history
      layer.weight_cache = np.zeros_like(layer.weights) # Store history of previous updates to use with momentum (moving average of the cache)
      layer.bias_momentums = np.zeros_like(layer.biases)
      layer.bias_cache = np.zeros_like(layer.biases)

    # Update momentum with current gradients (helps optimizer continue moving in the direction of previous updates)
    layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.d_weights
    layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.d_biases

    # Get corrected momentum (ensure updates are not overly influenced by initial zero bias)
    weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1)) # Add 1 to avoid division by zero
    bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

    # Update cache with squared current gradients (helps optimizer adapt LR for each weight based on history of its squared gradient updates)
    layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.d_weights**2
    layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.d_biases**2

    # Get corrected cache (lets less-frequently updated parameters keep-up with changes)
    weight_cache_corrected = layer.weight_cache / (1 - self.beta_2**(self.iterations + 1))
    bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.iterations + 1))

    # Parameter update + normalization with square rooted cache (weight_cache_corrected)
    layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
    layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

  # Increment the iteration counter
  def post_update_params(self):
    self.iterations += 1


############################## SGD Optimizer Class ##############################

# Stochastic Gradient Descent
class SGDOptimizer:

  # Initialize optimizer (learning rate of 1.0 is default for SGD)
  def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):

    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.momentum = momentum

  # Call once before any parameter updates
  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

  # Update parameters
  def update_params(self, layer):

    # If momentum is set
    if self.momentum:
      # Create zero matrices of weight/bias shape if layer does not have momentum arrays
      if not hasattr(layer, 'weight_momentums'): 
        layer.weight_momentums = np.zeros_like(layer.weights)
        layer.bias_momentums = np.zeros_like(layer.biases)
      # Build weight/bias updates with momentum
      weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.d_weights
      layer.weight_momentums = weight_updates
      bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.d_biases
      layer.bias_momentums = bias_updates
    # Vanilla SGD updates (same as before momentum update)
    else:
      weight_updates = -self.current_learning_rate * layer.d_weights
      bias_updates = -self.current_learning_rate * layer.d_biases

    # Update weights and biases using either vanilla or momentum updates
    layer.weights += weight_updates
    layer.biases += bias_updates

  # Call once after any parameter updates
  def post_update_params(self):
    self.iterations += 1


############################## Adagrad Optimizer Class ##############################

# Adaptive gradient
class AdagradOptimizer:

  # Initialize optimizer
  def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):

    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon

  # Call once before any parameter updates
  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

  # Update parameters
  def update_params(self, layer):
    # Create zero matrices of weight/bias shape if layer does not have cache arrays
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)

    # Update cache with squared current gradients
    layer.weight_cache += layer.d_weights**2
    layer.bias_cache += layer.d_biases**2

    # Vanilla SGD parameter update + normalization with square rooted cache
    layer.weights += -self.current_learning_rate * layer.d_weights / (np.sqrt(layer.weight_cache) + self.epsilon)
    layer.biases += -self.current_learning_rate * layer.d_biases / (np.sqrt(layer.bias_cache) + self.epsilon)

  # Call once after any parameter updates
  def post_update_params(self):
    self.iterations += 1


############################## RMSprop Optimizer Class ##############################

# Root Mean Square Propagation
class RMSpropOptimizer:

  # Initialize optimizer
  def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):

    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
    self.rho = rho

  # Call once before any parameter updates
  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

  # Update parameters
  def update_params(self, layer):
    # Create zero matrices of weight/bias shape if layer does not have cache arrays
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)

    # Update cache with squared current gradients
    layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.d_weights**2
    layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.d_biases**2

    # Vanilla SGD parameter update + normalization with square rooted cache
    layer.weights += -self.current_learning_rate * layer.d_weights / (np.sqrt(layer.weight_cache) + self.epsilon)
    layer.biases += -self.current_learning_rate * layer.d_biases / (np.sqrt(layer.bias_cache) + self.epsilon)

  # Call once after any parameter updates
  def post_update_params(self):
    self.iterations += 1    


############################## Loss Class ##############################

# Calculate data and regularization losses 
class Loss:
  
  # The regularization penalty for each dense layer is added to the overall loss
  def regularization_loss(self):

    # 0 by default
    regularization_loss = 0

    # Calculate entire model's regularization loss by iterating through all trainable layers
    for layer in self.trainable_layers:

      # L1 regularization (weights) - calculate only for factors greater than 0
      if layer.l1_w_reg > 0:
        # Add a penalty proportional to the sum of the absolute values of the weights
        regularization_loss += layer.l1_w_reg * np.sum(np.abs(layer.weights))

      # L2 regularization (weights)
      if layer.l2_w_reg > 0:
        # Add a penalty proportional to the sum of the squared values of the weights
        regularization_loss += layer.l2_w_reg * np.sum(layer.weights * layer.weights)

      # L1 regularization (biases)
      if layer.l1_b_reg > 0:
        regularization_loss += layer.l1_b_reg * np.sum(np.abs(layer.biases))

      # L2 regularization (biases)
      if layer.l2_b_reg > 0:
        regularization_loss += layer.l2_b_reg * np.sum(layer.biases * layer.biases)

    # Return the total calculated regularization loss to be added to data_loss during training
    return regularization_loss

  # Set/remember what layers are trainable before for-loop (updated in class Model finalize method)
  def set_trainable_layers(self, trainable_layers):
    self.trainable_layers = trainable_layers

  # Calculates data losses given model output (between prediction from specific loss function and ground truth values)
  def calculate_loss(self, output, y, *, include_regularization=False): # Set include_regularization=True for trainables def train method

    # Calculate sample losses (call the forward method of the specific loss function)
    sample_losses = self.forward(output, y)
    # Calculate mean loss
    data_loss = np.mean(sample_losses)

    # Add accumulated sum of losses and sample count to calculate a sample-wise average below
    self.accumulated_sum += np.sum(sample_losses) # Accumulate sum of losses from all epoch batches
    self.accumulated_count += len(sample_losses) # Calculate mean value at end of each epoch

    # Return only data loss for validation data (i.e. include_regularization=False)
    if not include_regularization:
      return data_loss

    # Return loss and regularization losses
    return data_loss, self.regularization_loss()

  # Calculates accumulated loss
  def accumulated_loss(self, *, include_regularization=False):

    # Calculate mean loss
    data_loss = self.accumulated_sum / self.accumulated_count

    # Only return data loss for validation
    if not include_regularization:
      return data_loss

    # Return the data and regularization losses
    return data_loss, self.regularization_loss()

  # Reset variables for accumulated loss (to calculate loss per epoch)
  def reset_accumulated_loss(self):
    self.accumulated_sum = 0
    self.accumulated_count = 0


############################## Categorical Cross-entropy Loss Class ##############################

# Inherits calculate function from Loss class (within training loop of Model class) when set in compilation
class CategoricalCrossentropyLoss(Loss):

# NB: NO BACKPROPAGATION USED - see combined Softmax Activation/Categorical Cross-entropy loss in Softmax classifier Class.

  # Performs error calculations and returns an object
  def forward(self, y_pred, y_true):

    # Get number of samples in a batch
    samples = len(y_pred)
    # Clip sample arrays of predicted probabilities to prevent 0 division and to avoid overall bias loss towards 1
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Probabilities for target values given label encoded (sparse) labels
    if len(y_true.shape) == 1: # See if target is single-dimensional
      correct_confidences = y_pred_clipped[range(samples), y_true] # one-hot encode labels

    # Mask values given one-hot encoded labels
    elif len(y_true.shape) == 2: # See if target is two-dimensional, e.g. [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
      correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # ...and mask labels [[0.7, 0., 0.], [0., 0.5, 0.], [0., 0.9, 0.]]

    # Losses (transform probabilities into a range of positive values)
    negative_log_likelihoods = -np.log(correct_confidences) # 0-1 probabilities have negative powers (10^(-1) = 0.1), given log curve
    return negative_log_likelihoods
  

############################## Softmax Classifier Class ##############################

# Combined backpropagation of Softmax Activation and Categorical Cross-entropy loss
class SoftmaxClassifier():

  # Backward pass
  def backward(self, d_values, y_true):

    # Get number of samples (from Catagorical Cross-entropy output layer)
    samples = len(d_values)

    # If labels are one-hot encoded (e.g. in preprocessing) turn them into a 1D array
    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)

    # Copy the gradients of the loss function w.r.t. the inputs of the combined functions
    self.d_inputs = d_values.copy()
    # Calculate gradient penalizing the model if predicted probability for the true class is less than 1
    self.d_inputs[range(samples), y_true] -= 1
    # Normalize gradient
    self.d_inputs = self.d_inputs / samples


############################## Binary Cross-entropy Loss Class ##############################

# Inherits calculate function from Loss class (within training loop of Model class) when set in compilation
class BinaryCrossentropyLoss(Loss):

  # Performs error calculations and returns an object
  def forward(self, y_pred, y_true):

   # Clip sample arrays of predicted probabilities to prevent 0 division and to avoid draging mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Calculate sample-wise loss (sum the log-likelihoods of the correct and incorrect classes for each neuron separately)
    sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    sample_losses = np.mean(sample_losses, axis=-1) # Calculate the mean value along the last dimension

    # Return losses
    return sample_losses

  # Backward pass
  def backward(self, d_values, y_true):

    # Get number of samples in a batch
    samples = len(d_values)
    # Use first sample to count number of outputs per sample
    outputs = len(d_values[0])
    # Clip sample arrays of predicted probabilities to prevent 0 division and to avoid draging mean towards any value
    d_values_clipped = np.clip(d_values_clipped, 1e-7, 1 - 1e-7)
    # Calculate gradient (the partial derivative of a sample loss w.r.t a single output loss)
    self.d_inputs = -(y_true / d_values_clipped - (1 - y_true) / (1 - d_values_clipped)) / outputs
    # Normalize gradient so it is invariant to the number of samples
    self.d_inputs = self.d_inputs / samples


############################## Mean Squared Error Loss Class (L2) ##############################
    
# Inherits calculate function from Loss class (within training loop of Model class) when set in compilation
class MeanSquaredErrorLoss(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

      # Calculate loss - axis=-1 calculates each sample mean across outputs
      sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
      # Return losses
      return sample_losses

    # Backward pass
    def backward(self, d_values, y_true):

      # Number of samples
      samples = len(d_values)
      # Use the first sample to count number of outputs in every sample
      outputs = len(d_values[0])

      # Gradient on values
      self.d_inputs = -2 * (y_true - d_values) / outputs
      # Normalize gradient by samples to make them invariant to the batch size
      self.d_inputs = self.d_inputs / samples
      

############################## Mean Absolute Error Loss Class (L1) ##############################

# Inherits calculate function from Loss class (within training loop of Model class) when set in compilation
class MeanAbsoluteErrorLoss(Loss):

  # Forward pass
  def forward(self, y_pred, y_true):

    # Calculate loss - axis=-1 calculates each sample mean across outputs
    sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
    # Return losses
    return sample_losses

  # Backward pass
  def backward(self, d_values, y_true):

    # Number of samples
    samples = len(d_values)
    # Use the first sample to count number of outputs in every sample
    outputs = len(d_values[0])

    # Calculate gradient - np.sign returns 1 or -1 given the sign of the input and 0 if the parameter equals 0
    self.d_inputs = np.sign(y_true - d_values) / outputs
    # Normalize gradient by samples to make them invariant to the batch size
    self.d_inputs = self.d_inputs / samples
    
    
############################## Accuracy Class ##############################

class Accuracy:

  # Calculates accuracy between predictions (from activation classes) and ground truth values
  def calculate_accuracy(self, predictions, y):

    # Get comparison results from classification or regression child class
    comparisons = self.compare(predictions, y)# returns a list of True and False values (see child)
    # Calculate accuracy (treats True as 1 and False as 0)
    accuracy = np.mean(comparisons)

    # Add accumulated sum of matching values and sample count
    self.accumulated_sum += np.sum(comparisons) # Accumulate sum of comparisons from all epoch batches
    self.accumulated_count += len(comparisons) # Calculate mean value at end of each epoch

    # Return accuracy
    return accuracy

  # Calculates accumulated accuracy
  def accumulated_accuracy(self):
    # Calculate an accuracy
    accuracy = self.accumulated_sum / self.accumulated_count
    # Return the data and regularization losses
    return accuracy

  # Reset variables for accumulated accuracy (to calculate accuracy per epoch)
  def reset_accumulated_accuracy(self):
    self.accumulated_sum = 0
    self.accumulated_count = 0
    

############################## Classification Accuracy Class ##############################

class ClassificationAccuracy(Accuracy):

  # Not used but exists for Model class to call to calculate 'precision' in case of RegressionAccuracy
  def init_accuracy(self, y):
    pass # Do nothing

  # Compares predictions to the ground truth values
  def compare(self, predictions, y):

    # If labels are one-hot encoded, turn them into a 1D array to account for binary predictions
    if len(y.shape) == 2:
      y = np.argmax(y, axis=1)
    return predictions == y # Returns a list of True and False values to be used in parent class
      

############################## Regression Accuracy Class ##############################

class RegressionAccuracy(Accuracy):

  # Initialize the precision attribute to calculate the accuracy of regression models
  def __init__(self):
    # Create precision property
    self.precision = None # There are no predictions when object is created

  # Calculates precision value based on passed in ground truth values
  def init_accuracy(self, y, re_init=False): # Calculates average accuracy called from by Model class 

    # Only recalculate precision if re_init=True (resets accumulated accuracy and calculates for current epoch only vs re_init=False)
    if self.precision is None or re_init:
      self.precision = np.std(y) / 250

  # Compares predictions to the ground truth values
  def compare(self, predictions, y):
    return np.absolute(predictions - y) < self.precision # Returns a list of True and False values to be used in parent class
  

############################## Model Class ##############################

class Model:

  # Called when a new object model is created
  def __init__(self):

    # Create a list of neural network objects (see add() method below)
    self.layers = []
    # Initialize Softmax classifier's output object to keep tack of combined operation (see def finalize below)
    self.softmax_classifier_output = None # Softmax classifier has no forward pass to calculate and store outputs

  # Add objects to the model (above list)
  def add(self, layer):
    self.layers.append(layer)

  # Set loss, optimizer, accuracy (to compilation/model.set() method)
  def set(self, *, loss=None, optimizer=None, accuracy=None):# (*) -> explicitly use parameter names ('loss=' etc.)
    # 'None' added to allow for passing of given parameters (set_parameters() method from class DenseLayer)
    if loss is not None:
      self.loss = loss
    if optimizer is not None:
      self.optimizer = optimizer
    if accuracy is not None:
      self.accuracy = accuracy

  # Finalize the model - set previous and next layer properties for each layer
  def finalize(self):

    # Create and set the input layer (no weights and biases)
    self.input_layer = InputLayer() # necessary for first layer in below iteration
    # Count the number of layers that have been added to the model
    layer_count = len(self.layers)
    # Initialize a list containing trainable layers (see hasattr() check below)
    self.trainable_layers = []

    # Iterate through layers and classify trainables
    for i in range(layer_count):

      # The first layer has input_layer as it's previous layer
      if i == 0:
        self.layers[i].prev = self.input_layer
        self.layers[i].next = self.layers[i + 1]

      # Store layer outputs for all intermediate layers
      elif layer_count - 1 > i: # layer_count minus input_layer
        self.layers[i].prev = self.layers[i - 1] 
        self.layers[i].next = self.layers[i + 1]

      # The last layer has the loss layer (model output) as next layer
      else:
        self.layers[i].prev = self.layers[i - 1]# Index of the layer before it
        self.layers[i].next = self.loss
        self.output_layer_activation = self.layers[i] # Save reference to last layer's output for activation predictions in train method

      # Add layers with 'weights' attribute to above list of trainable layers (non-validation layers)
      if hasattr(self.layers[i], 'weights'):
        self.trainable_layers.append(self.layers[i])

      # Update class Loss with trainable layers (for regularization calculations)
      if self.loss is not None: # Only if this loss object exists, given no training for set_parameters
        self.loss.set_trainable_layers(self.trainable_layers)

    # Given Softmax activation output AND Categorical Cross-Entropy loss create combined activation/loss object
    if (isinstance(self.layers[-1], SoftmaxActivation) and # Softmax is the last layer
        isinstance(self.loss, CategoricalCrossentropyLoss)): # The loss function should be a CategoricalCrossentropy object
       # Create an object of combined activation and loss functions
       self.softmax_classifier_output = SoftmaxClassifier()

  # Train the model (use entire dataset as the batch if batch_size=None)
  def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

    # Initialize accuracy with true labels (see def set() method above)
    self.accuracy.init_accuracy(y)# Initializes/resets RegressionAccuracy precision method

    # Given batch_size=None take 1 step per epoch (feed entire dataset at each epoch)
    train_steps = 1

    # Calculate number of steps if batch size is given
    if batch_size is not None:
      train_steps = len(X) // batch_size

      # Add 1 for remainder after division round-off (given integer division operator)
      if train_steps * batch_size < len(X):
        train_steps += 1

      # Main training loop
      for epoch in range(1, epochs + 1):

        # Print epoch number
        print(f'epoch: {epoch}')

        # Reset accumulated values in loss and accuracy for new training batches
        self.loss.reset_accumulated_loss()
        self.accuracy.reset_accumulated_accuracy()

        # Iterate over training data steps (for each epoch)
        for step in range(train_steps):
          # If batch size is not set, use one step for full dataset
          if batch_size is None:
            batch_X = X
            batch_y = y
          # Otherwise grab a batch of data to train (slice of size batch_size)
          else:
            batch_X = X[step * batch_size : (step + 1) * batch_size]# Do one step at a time [step : (step + 1)]
            batch_y = y[step * batch_size : (step + 1) * batch_size]

          # Perform forward pass on training data batches (see forward() method below)
          output = self.forward(batch_X, training=True) # Pass data batches and set training parameter in forward method to True
          # Calculate loss (monitor regularization contribution to the overall loss)
          data_loss, regularization_loss = self.loss.calculate_loss(output, batch_y, include_regularization=True)# False as default in class Loss
          loss = data_loss + regularization_loss
          # Get predictions (congruent to output layer's activation: softmax, sigmoid, linear) and calculate accuracy
          predictions = self.output_layer_activation.predictions(output)# Last layer output referenced in finalize method
          accuracy = self.accuracy.calculate_accuracy(predictions, batch_y)

          # Perform backward pass on training data (see backward() method below)
          self.backward(output, batch_y)
          # Call optimizer
          self.optimizer.pre_update_params() # Calculates the adjusted LR if a decay factor is specified
          for layer in self.trainable_layers: # self.trainable_layers is set in finalize() method
            self.optimizer.update_params(layer) # Uses adjusted LR to update the model's parameters w/momentum and cache
          self.optimizer.post_update_params() # Increments the counter

          # Print a summary per training data epoch train_steps
          if not step % print_every or step == train_steps - 1:
            print(f'step: {step}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f} (' +
                  f'data_loss: {data_loss:.3f}, ' +
                  f'reg_loss: {regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

        # Get accumulated/total loss and accuracy on training data
        epoch_data_loss, epoch_regularization_loss = self.loss.accumulated_loss(include_regularization=True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.accumulated_accuracy()

        # Print accumulated/total summary on training data
        print(f'training, ' +
              f'acc: {epoch_accuracy:.3f}, ' +
              f'loss: {epoch_loss:.3f} (' +
              f'data_loss: {epoch_data_loss:.3f}, ' +
              f'reg_loss: {epoch_regularization_loss:.3f}), ' +
              f'lr: {self.optimizer.current_learning_rate}')

        # Add batching for the validation data if batch_size is set
        if validation_data is not None:
          # Evaluate the model
          self.evaluate(*validation_data, batch_size=batch_size)# '*' unpacks validation_data list into singular values

  # Evaluates the model
  def evaluate(self, X_val, y_val, *, batch_size=None):

    # Default value if batch size is not being set
    validation_steps = 1

    # Calculate number of steps
    if batch_size is not None:
      validation_steps = len(X_val) // batch_size

      # Add 1 for remainder after division round-off
      if validation_steps * batch_size < len(X_val):
        validation_steps += 1

    # Reset accumulated values in loss and accuracy for new validation batches
    self.loss.reset_accumulated_loss()
    self.accuracy.reset_accumulated_accuracy()

    # Iterate over validation data steps (for each epoch)
    for step in range(validation_steps):

      # If batch size is not set, use one step for full dataset
      if batch_size is None:
        batch_X = X_val
        batch_y = y_val
      # Otherwise grab a batch of data to validate (slice of size batch_size)
      else:
        batch_X = X_val[step * batch_size : (step + 1) * batch_size]
        batch_y = y_val[step * batch_size : (step + 1) * batch_size]

      # Perform only forward on validation data batches (see method below)
      output = self.forward(batch_X, training=False)
      # Calculate the loss to be accumulated below (Batch losses are accumulated in 'def calculate_loss' and passed to 'def accumulated_loss')
      self.loss.calculate_loss(output, batch_y) # Not stored, since only accumulated is printet
      # Get predictions and calculate accuracy to be accumulated below
      predictions = self.output_layer_activation.predictions(output)
      self.accuracy.calculate_accuracy(predictions, batch_y) # Not stored, since only accumulated is printet

    # Get accumulated/total loss and accuracy on validation data
    validation_loss = self.loss.accumulated_loss()
    validation_accuracy = self.accuracy.accumulated_accuracy()

    # Print accumulated/total summary on validation data
    print(f'validation, ' +
          f'acc: {validation_accuracy:.3f}, ' +
          f'loss: {validation_loss:.3f}')

  def predict(self, X, *, batch_size=None):# predict X with possibility of passing a batch

    # Default value if batch size is not being set
    prediction_steps = 1

    # Calculate number of steps similar to the evaluate method
    if batch_size is not None:
      prediction_steps = len(X) // batch_size

      # Add 1 for remainder after division round-off
      if prediction_steps * batch_size < len(X):
        prediction_steps += 1

    # List of model output 
    output = []

    # Iterate over steps
    for step in range(prediction_steps):
      # If batch size is not set, use one step for full dataset
      if batch_size is None:
        batch_X = X
      # Otherwise slice a batch
      else:
        batch_X = X[step * batch_size : (step + 1) * batch_size]

      # Perform the forward pass
      batch_output = self.forward(batch_X, training=False)
      # Append batch prediction to the list of predictions
      output.append(batch_output)

    # Stack and return results
    return np.vstack(output)

  # Forward pass for the entire neural network (vs. other classes specific/low-level forward computations)
  def forward(self, X, training): # training (boolean) is called as True/False in train method

    # Call forward on input_layer to set output to first 'prev' layer
    self.input_layer.forward(X, training) # InputLayer object created in finalize method

    # Call forward on every intermediate layers
    for layer in self.layers: # Initialized in __init__ method with outputs stored in finalize method
      layer.forward(layer.prev.output, training) # Pass output of the previous layer as a parameter

    # Return output of last layer from the list
    return layer.output

  # Backward pass
  def backward(self, output, y):

    # Backpropagation given activation of softmax classifier
    if self.softmax_classifier_output is not None: # From finalize() method
      # Call backward method on combined activation/loss to set d_inputs property
      self.softmax_classifier_output.backward(output, y)
      # Since we used combined activation/loss, don't call last layer (Softmax) but set d_inputs in this object
      self.layers[-1].d_inputs = self.softmax_classifier_output.d_inputs # d_inputs from SoftmaxClassifier

      # Call backward method on all objects but last, passing above d_inputs in reversed order
      for layer in reversed(self.layers[:-1]):
        layer.backward(layer.next.d_inputs) # next from finalize() method

      return

    # Backpropagation given alternative loss functions (binary cross-entropy or mean squared error)
    self.loss.backward(output, y)

    # Call backward method on all objects passing d_inputs in reversed order
    for layer in reversed(self.layers):
      layer.backward(layer.next.d_inputs)

  # Retrieve and return trainable layers parameters
  def get_parameters(self):

    # Create parameters list
    parameters = []

    # Iterate over trainable layers and get their parameters
    for layer in self.trainable_layers:
      parameters.append(layer.get_parameters())
    # Return a list
    return parameters

  # Update model with new parameters
  def set_parameters(self, parameters):
    # Iterate over parameters and layers and update layers with each set of parameters
    for parameter_set, layer in zip(parameters, self.trainable_layers):# Return a new iterable with pairwise combinations of layers and their parameters
      layer.set_parameters(*parameter_set)# '*' unpacks tuple of weights and biases into singular values

  # Save parameters to a file w/necessary information to recreate the object later
  def save_parameters(self, path):
    # Open a file in binary-write mode 'wb' and save parameters into it
    with open(path, 'wb') as f:
      pickle.dump(self.get_parameters(), f)# Serialize into a binary representation w/pickle

  # Load weights and update a model instance with them
  def load_parameters(self, path):
    # Open file in binary-read mode 'rb', load weights and update trainable layers
    with open(path, 'rb') as f:
      self.set_parameters(pickle.load(f)) # Deserialize loaded parameters back into a list

  # Save the model
  def save(self, path):

    # Make a deep copy (recursively traverse all objects) of current model instance
    model = copy.deepcopy(self)

    # Reset accumulated values in loss and accuracy objects (only track from the point of reloading)
    model.loss.reset_accumulated_loss()
    model.accuracy.reset_accumulated_accuracy()

    # Clean up (remove data from) input layer and gradients from loss object
    model.input_layer.__dict__.pop('output', None) # The output attribute of input_layer is not part of the trained model itself
    model.loss.__dict__.pop('d_inputs', None) # Remove key-value pair of internal dictionary storing attributes of loss object

    # For each layer remove inputs, output and gradient properties (NB: d_inputs per layer vs d_inputs from loss object)
    for layer in model.layers: # Layer-specific properties, as opposed to above model-level properties associated with loss/input
      for property in ['inputs', 'output', 'd_inputs', 'd_weights', 'd_biases']:
        layer.__dict__.pop(property, None)

    # Open a file in the binary-write mode 'wb' and save the model
    with open(path, 'wb') as f:
      pickle.dump(model, f)

  # Load and return a model (use decorator to run class methods, since no access/modification of any object data is needed)
  @staticmethod
  def load(path):
    # Open file in the binary-read 'rb' mode and load a model
    with open(path, 'rb') as f:
      model = pickle.load(f) # deserialize the saved model

    # Return a model
    return model