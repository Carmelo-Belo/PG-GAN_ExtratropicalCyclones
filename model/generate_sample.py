from tf.keras.models import load_model
from PGGAN_architecture_comb import PixelNormalization
from PGGAN_data_functions import generate_examples

# Load the generator model and weights 

load_model = 'trained_generator/064x064-tuned.h5'
load_weights = 'trained_generator/model_weigths_064x064-tuned.h5'

generator = load_model(load_model, custom_objects={'PixelNormalization': PixelNormalization})
generator.load_weights(load_weights)

# Generate points in latent space as input for the generator

Z_DIM = 256
n_examples = 1000
gen_sample = generate_examples(generator, Z_DIM, n_examples)

# Set negative values of the generated samples to zero
gen_sample[gen_sample < 0] = 0