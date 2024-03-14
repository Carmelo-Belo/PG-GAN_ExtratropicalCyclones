import time
from PGGAN_architecture_comb import define_discriminator, define_generator, define_composite
from PGGAN_data_functions import load_real_samples
from PGGAN_training_functions import train

## TRAINING OF THE MODEL

# number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
n_blocks = 5

# learning rate
lr = 10e-5

# Adam optimizer hyperparameters
b_1 = 0.0
b_2 = 0.99
eps = 10e-8

# size of the latent space
latent_dim = 256

# number of batches for each growth phase
n_batch_res = [16, 16, 16, 8, 8]
n_batch_mul = 6
n_batch = [element * n_batch_mul for element in n_batch_res]

# number of epochs for each growth phase
n_epochs_res = [5, 10, 10, 15, 15]
n_epochs_mul = 3
n_epochs = [element * n_epochs_mul for element in n_epochs_res]

# Define models & input shape, the input shape depends on the training set
in_shape = (4, 4, 3)
channels = in_shape[-1]
d_models = define_discriminator(n_blocks, lr, b_1, b_2, eps, input_shape=in_shape)
g_models = define_generator(latent_dim, n_blocks, channels)

# Define composite models
gan_models = define_composite(d_models, g_models, lr, b_1, b_2, eps)

# load training data
# Here the training set need to be imported, eventual data transformation needs to be added by user
train_file = 'data/training/train_set.npy'
dataset = load_real_samples(train_file)
print(f'Original dataset dimensions: {dataset.shape}')
print('Loaded', dataset.shape)

# Save folder definition for where we want to save model and weights at the end of training
save_folder = f'run_{time.strftime("%Y%m%d-%H%M", time.localtime())}/'

# train the model
start_time = time.time()
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch, save_folder)
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

# Save highest resolution tuned generator and discriminator model architecture
g_model = g_models[-1][0]
gen_shape = g_model.output_shape
name = '%03dx%03d-tuned.h5' % (gen_shape[1], gen_shape[2])
g_model.save(save_folder+name)
d_model = d_models[-1][0]
d_name = 'disc_%03dx%03d-tuned.h5' % (gen_shape[1], gen_shape[2])
d_model.save(save_folder+d_name)