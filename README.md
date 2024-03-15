# Synthetic Generation of Extra-Tropical Cyclones’ Fields with Generative Adversarial Networks

## Overview 

This repository contains the code data necessary to replicate the findings and experiments presented in our paper, "Synthetic Generation of Extra-Tropical Cyclones’ fields with Generative Adversarial Networks". The work introduces a deep learning approach to generating realistic, synthetic extra-tropical cyclone atmospheric fields using the Progressive Growing Generative Adversarial Network (PG-GAN), and ERA5 reanalysis data and historical cyclone tracks to train the network.
To download the data we used to train and evaluate the network please refer to the repository on zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10821921.svg)](https://doi.org/10.5281/zenodo.10821921)

Please refer to the [](link paper) and [](link supporting information) for more details regarding the project.

## Repository Structure, Data and Model

This repository is organized into folders and files to facilitate easy replication of our work and experimentation with the PG-GAN. Below is an overview of the structure and contents:

### model directory

The **'model'** directory is the core of this repository, containing all the necessary scripts and resources to train the GAN model from scratch, generate new samples, and understand the architecture behind the network. Here’s what you can find inside:

- **PGGAN_architecture_comb.py**: Contains the functions to define the architecture of the PG-GAN network used in our study.
- **PGGAN_data_function.py**: Provides the functions required for processing the data for output from the network.
- **PGGAN_training_functions.py**: Includes the functions for training the model. It contains the core training loop, optimization algorithms, and other utilities needed to train the PG-GAN.
- **train_PGGAN.py**: This script is used to launch the training process of the model. Within this file, users can adjust the training hyperparameters to experiment with different configurations. It integrates the above scripts for a complete training setup.
- **generate_sample.py**: Allows users to generate new synthetic samples using the trained network. This script utilizes the trained generator to produce new data points.

Subfolders:

- **data**: Contains the dataset used for training as well as the three datasets used to evaluate the network's performance. It’s essential for users looking to replicate our results or understand how the network performs with different data. The file ***training/train_set.npy*** contains the fields of the three variables fields stacked a channel dimension (0) mean sea level pressure (1) wind speed at 10m (2) rainfall rate. The mean sea level pressure is normalized according to the historical maximum and minimum ever recorded in the North Atlantic, respectively 1057 hPa and 913 hPa. The wind speed and rainfall rate are normalized by logarithmic normalization. 
This folder contains large data files (> 5GB), it is available on zenodo.
- **trained_generator**: This subfolder houses the architecture and weights of the generator used in our study. It allows users to generate new cyclone fields without needing to train the model from scratch, facilitating easy experimentation and further research.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following prerequisites installed:

- Python >= 3.11
- Numpy >= 1.26
- Scikit Image >= 0.22
- Tensorflow >= 2.15

### Installation

1. Clone the repository:

```bash
    git clone https://github.com/Carmelo-Belo/PG-GAN_ExtratropicalCyclones
```

2. Install the required dependecies in your conda environment:

```bash
    conda install numpy scikit tensorflow
```

### Utilizing the repository

To get started with training the model or generating new samples, navigate to the **'model directory'**. Here, you can run ***train_PGGAN.py*** to train your model with the provided or your data. Adjusting the hyperparameters within this file will let you experiment with the network's performance and efficiency.

For generating new synthetic cyclone fields, use ***generate_sample.py*** with the trained generator found in the ***trained_generator*** subfolder. This allows for immediate generation of new data points, useful for further analysis, comparison, or integration into climate models.

## Citation

If you use our code, data, or findings in your work, please cite our paper:

(insert citation)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries regarding this work, please contact:

- Filippo Dainelli: [filippo.dainelli@polimi.it]
