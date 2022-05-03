# CS8395 DL HW4
This repo contains all the code for CS8395 DL HW4 and running instructions which will help TA and instructor to execute the code

### Helper modules
* _utils.py_: Contains all the helper functions
* _models.py_: Contains RotNet and CNN models
* _pre\_train\_rotnet.py_: A script to train rotnet model which would act as pretrain model for the mnist classifier.
* _train\_mnist.py_: script to train the mnist classifier using pre-trained rotnet model
* _gen\_adv\_examples.py_: Script to generate adv sampled by deploying adv attacks on the models.


## How to execute the code
### Pre-training
To execute the training script you can simply run _pre\_train\_rotnet.py_. Which would be:
```bash
python3 _pre_train_rotnet.py
```
### MNIST Training
To execute the training script you can simply run _train\_mnist.py_. It will train cnn model taking rotnet as pretrained model
```bash
python3 train_mnist.py
```
### Generation of Adv Samples
To deploy adversarial attacks and generated perturbated samples you can simply run _gen\_adv\_examples.py_. It will generate both targeted and untargeted samples.
```bash
python3 gen_adv_examples.py
```
