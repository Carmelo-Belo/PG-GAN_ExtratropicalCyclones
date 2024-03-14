from skimage.transform import resize
from numpy import load, ones, exp, stack, asarray
from numpy.random import randn, randint

## Functions to load, eventually modify the training dataset, and to create the samples
# load dataset
def load_real_samples(filename):
	# load dataset
	X = load(filename)
	# convert from ints to floats
	X = X.astype('float32')
    # Data are already normalized, but you could do it here instead
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y

## Functions to create n examples from the latent space for the generator
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input, verbose='0')
	# create class labels
	y = -ones((n_samples, 1))
	return X, y

# Function to resize a numpy array
def resize_array(x, new_rows, new_cols): 
    # Order of interpolation
    # 0: Nearest-neighbor, 1: Bi-linear (default), 2: Bi-quadratic, 3: Bi-cubic, 4: Bi-quartic, 5: Bi-quintic
    output_shape = (new_rows, new_cols)
    # anti_aliasing -> Whether to apply a Gaussian filter to smooth the image prior to downsampling. 
    # It is crucial to filter when downsampling the image to avoid aliasing artifacts
    output = resize(x, output_shape, order=1, mode='reflect', anti_aliasing=True)
    return output

# Class to denormalize and reshape the output of the generator model
class OutputProcess:
    def __init__(self, output_shape=(201,481)):
        self.output_shape = output_shape
        # Define denormalization parameters for each channel
        self.denorm_params = (1057, 913)   
    
    def forward(self, arr):
        # Denormalize each channel
        for i in range(3):
            max_val, min_val = self.denorm_params
            channel = arr[:,:,i]
            if i == 0:
                channel = channel * (max_val - min_val) + min_val
            else:
                channel = exp(channel) - 1
            arr[:,:,i] = channel
        # Resize array to input shape
        channel1 = resize_array(arr[:,:,0], self.output_shape[0], self.output_shape[1])
        channel2 = resize_array(arr[:,:,1], self.output_shape[0], self.output_shape[1])
        channel3 = resize_array(arr[:,:,2], self.output_shape[0], self.output_shape[1])
        arr = stack((channel1,channel2,channel3), axis=2)
        
        return arr
    
# denormalize and resize generated sample
def generate_examples(generator, latent_dim, n_samples):
    output = generate_fake_samples(generator, latent_dim, n_samples)
    # denormalize and reshape output
    transform = OutputProcess()
    sample = []
    for ex, example in enumerate(output):
        field = transform.forward(example)
        sample.append(field)
    return asarray(sample)

