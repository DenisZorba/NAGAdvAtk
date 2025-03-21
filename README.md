## Adversarial Attack Using General-Purpose Optimisers via the NAG Library

This repository contains code that performs both targeted and untargeted adversarial attacks on image classifiers by leveraling the general-purpose optimisers (**IPOPT** and **SSQP**) provided by the Numerical Analysis Group (NAG)


| **Hamster (Original)** | **Gas Pump (Attacked)**  | 
| :------------: | :--------------: | 
| ![Hamster (original)](example_img/hamsterToGasPump_org.png) | ![Gas Pump (attacked)](example_img/hamsterToGasPump_atked.png) | 

## Prerequisites

- **Dependencies:** The code was tested on `naginterfaces` version **29.0.0** and `pytorch` version **2.5.0**. Other versions might also work.
- **Installation Instructions:** To install the NAG library, please follow the [official installation guide](https://support.nag.com/numeric/py/nagdoc_latest/readme.html).


## Usage

### Adversarial attack on MNIST
The MNIST attack script performs attacks on the `lenet_mnist_model.pth` classifier. To run it, simply execute:
```bash
python minimise_dx_norm_MNIST.py
```

### Adversarial attack on ImageNet
For running adversarial attacks on ImageNet iamges, follow these steps:

1. Download the ImageNet validation set from [Academic Torrents](https://academictorrents.com/browse.php?search=imagenet&sort_field=seeders&sort_dir=DESC)
2. Open `minimise_dx_norm_MNIST.py` and replace the existing path='...' on line 19 with the extracted file path to the ImageNet validation set
3. Execute the script 
```bash
Python minimise_dx_norm_MNIST.py
```

