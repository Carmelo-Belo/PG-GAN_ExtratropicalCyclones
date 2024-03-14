from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Layer, Add
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend

## Definition of the three Custom Layer for the Progressive Growing GAN:
## WeightedSum: used to control the weighted sum of the old and new layers during a growth phase
## MinibatchStdev: used to summarize statistics for a batch of images in the discriminator
## PixelNormalization: used to normalize activation maps in the generator model weighted sum output

class WeightedSum(Add):
  # init with default value
  def __init__(self, alpha=0.0, **kwargs):
    super(WeightedSum, self).__init__(**kwargs)
    self.alpha = backend.variable(alpha, name='ws_alpha')

  # output a weighted sum of inputs
  def _merge_function(self, inputs):
    # only supports a weighted sum of two inputs
    assert (len(inputs) == 2)
    # ((1-a) * input1) + (a * input2)
    output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
    return output

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'alpha': self.alpha,
    })
    return config

class MinibatchStdev(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(MinibatchStdev, self).__init__(**kwargs)

	# perform the operation
	def call(self, inputs):
		# calculate the mean value for each pixel across channels
		mean = backend.mean(inputs, axis=0, keepdims=True)
		# calculate the squared differences between pixel values and mean
		squ_diffs = backend.square(inputs - mean)
		# calculate the average of the squared differences (variance)
		mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
		# add a small value to avoid a blow-up when we calculate stdev
		mean_sq_diff += 1e-8
		# square root of the variance (stdev)
		stdev = backend.sqrt(mean_sq_diff)
		# calculate the mean standard deviation across each pixel coord
		mean_pix = backend.mean(stdev, keepdims=True)
		# scale this up to be the size of one input feature map for each sample
		shape = backend.shape(inputs)
		output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
		# concatenate with the output
		combined = backend.concatenate([inputs, output], axis=-1)
		return combined

	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		# create a copy of the input shape as a list
		input_shape = list(input_shape)
		# add one to the channel dimension (assume channels-last)
		input_shape[-1] += 1
		# convert list to a tuple
		return tuple(input_shape)

# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(PixelNormalization, self).__init__(**kwargs)

	# perform the operation
	def call(self, inputs):
		# calculate square pixel values
		values = inputs**2.0
		# calculate the mean pixel values
		mean_values = backend.mean(values, axis=-1, keepdims=True)
		# ensure the mean is not zero
		mean_values += 1.0e-8
		# calculate the sqrt of the mean squared value (L2 norm)
		l2 = backend.sqrt(mean_values)
		# normalize values by the l2 norm
		normalized = inputs / l2
		return normalized

	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		return input_shape

## Definition of the cramer loss which is used to train the model
# calculate cramer loss
def cramer_loss(y_true, y_pred):
	return backend.sum((y_true - y_pred)**2)

## Definition of the discriminator architecture, two functions are needed,
## one to add a new block of layers in the model, the other to define it for the
## different images resolution
# add a discriminator block
def add_discriminator_block(old_model, n_input_layers=3, lr=10e-5, b_1=0.0, b_2=0.99, eps=10e-8):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # weight constraint
  const = max_norm(1.0)
  # get shape of existing model
  in_shape = list(old_model.input.shape)
  # define new input shape as double the size
  # input_shape = (in_shape[-2].value*2, in_shape[-2].value*2, in_shape[-1].value)
  input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
  in_image = Input(shape=input_shape)
  # define new input processing layer
  d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  # define new block
  d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = AveragePooling2D()(d)
  block_new = d
  # skip the input, 1x1 and activation for the old model
  for i in range(n_input_layers, len(old_model.layers)):
    d = old_model.layers[i](d)
  # define straight-through model
  model1 = Model(in_image, d)
  # compile model
  model1.compile(loss=cramer_loss, optimizer=Adam(learning_rate=lr, beta_1=b_1, beta_2=b_2, epsilon=eps))
  # downsample the new larger image
  downsample = AveragePooling2D()(in_image)
  # connect old input processing to downsampled new input
  block_old = old_model.layers[1](downsample)
  block_old = old_model.layers[2](block_old)
  # fade in output of old model input layer with new input
  d = WeightedSum()([block_old, block_new])
  # skip the input, 1x1 and activation for the old model
  for i in range(n_input_layers, len(old_model.layers)):
    d = old_model.layers[i](d)
  # define straight-through model
  model2 = Model(in_image, d)
  # compile model
  model2.compile(loss=cramer_loss, optimizer=Adam(learning_rate=lr, beta_1=b_1, beta_2=b_2, epsilon=eps))
  return [model1, model2]

# define the discriminator models for each image resolution
def define_discriminator(n_blocks, lr=10e-5, b_1=0.0, b_2=0.99, eps=10e-8, input_shape=(4,4,3)):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # weight constraint
  const = max_norm(1.0)
  model_list = list()
  # base model input
  in_image = Input(shape=input_shape)
  # conv 1x1
  d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  # conv 3x3 (output block)
  d = MinibatchStdev()(d)
  d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # conv 4x4
  d = Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # dense output layer
  d = Flatten()(d)
  out_class = Dense(1)(d)
  # define model
  model = Model(in_image, out_class)
  # compile model
  model.compile(loss=cramer_loss, optimizer=Adam(learning_rate=lr, beta_1=b_1, beta_2=b_2, epsilon=eps))
  # store model
  model_list.append([model, model])
  # create submodels
  for i in range(1, n_blocks):
    # get prior model without the fade-on
    old_model = model_list[i - 1][0]
    # create new model for next resolution
    models = add_discriminator_block(old_model, n_input_layers=3, lr=lr, b_1=b_1, b_2=b_2, eps=eps)
    # store model
    model_list.append(models)
  return model_list

## Generator architecture is defined in the same way as the discriminator
# add a generator block
def add_generator_block(old_model, channels):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = max_norm(1.0)
	# get the end of the last block
	block_end = old_model.layers[-2].output
	# upsample, and define new block
	upsampling = UpSampling2D()(block_end)
	g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
	g = PixelNormalization()(g)
	g = LeakyReLU(alpha=0.2)(g)
	g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
	g = PixelNormalization()(g)
	g = LeakyReLU(alpha=0.2)(g)
	# add new output layer
	out_image = Conv2D(channels, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
	# define model
	model1 = Model(old_model.input, out_image)
	# get the output layer from old model
	out_old = old_model.layers[-1]
	# connect the upsampling to the old output layer
	out_image2 = out_old(upsampling)
	# define new output image as the weighted sum of the old and new models
	merged = WeightedSum()([out_image2, out_image])
	# define model
	model2 = Model(old_model.input, merged)
	return [model1, model2]

# define generator models
def define_generator(latent_dim, n_blocks, channels, in_dim=4):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = max_norm(1.0)
	model_list = list()
	# base model latent input
	in_latent = Input(shape=(latent_dim,))
	# linear scale up to activation maps
	g  = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
	g = Reshape((in_dim, in_dim, 128))(g)
	# conv 4x4, input block
	g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
	g = PixelNormalization()(g)
	g = LeakyReLU(alpha=0.2)(g)
	# conv 3x3
	g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
	g = PixelNormalization()(g)
	g = LeakyReLU(alpha=0.2)(g)
	# conv 1x1, output block
	out_image = Conv2D(channels, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
	# define model
	model = Model(in_latent, out_image)
	# store model
	model_list.append([model, model])
	# create submodels
	for i in range(1, n_blocks):
		# get prior model without the fade-on
		old_model = model_list[i - 1][0]
		# create new model for next resolution
		models = add_generator_block(old_model, channels)
		# store model
		model_list.append(models)
	return model_list

## Definition of the composite model to train the generator and the discriminator together
# define composite models for training generators via discriminators
def define_composite(discriminators, generators, lr=10e-5, b_1=0.0, b_2=0.99, eps=10e-8):
	model_list = list()
	# create composite models
	for i in range(len(discriminators)):
		g_models, d_models = generators[i], discriminators[i]
		# straight-through model
		d_models[0].trainable = False
		model1 = Sequential()
		model1.add(g_models[0])
		model1.add(d_models[0])
		model1.compile(loss=cramer_loss, optimizer=Adam(learning_rate=lr, beta_1=b_1, beta_2=b_2, epsilon=eps))
		# fade-in model
		d_models[1].trainable = False
		model2 = Sequential()
		model2.add(g_models[1])
		model2.add(d_models[1])
		model2.compile(loss=cramer_loss, optimizer=Adam(learning_rate=lr, beta_1=b_1, beta_2=b_2, epsilon=eps))
		# store
		model_list.append([model1, model2])
	return model_list
