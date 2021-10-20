### Motivation
The goal of this project is to create a classifier that can identify 1st generation pokemon. Despite what the name suggests, this is not a classifier designed to play the "Who's that pokemon?" game from the anime, though images from that game will be tested later. As far as what the motivation behind this project is, I'm curious how a CNN will be able to adapt to images that are mainly hand drawn (specifically whether transfer learning will be less effective in this context) and how good performance will be with 151 different classes.

### File Structure
```
.
├── PokemonData                  | All the images used to train and validate every model.
│   ├── Test
│   │   ├── Abra
│   │   │   ├── .jpg
│   │   │   ├── .jpg
│   │   │   ├── ...
│   │   ├── Aerodactyl
│   │   ├── Alakazam
│   │   ├── ...
│   └── Train
│       ├── Abra
│       │   ├── .jpg
│       │   ├── .jpg
│       │   ├── ...
│       ├── Aerodactyl
│       ├── Alakazam
│       ├── ...
├── pokemon_classification.ipynb | All model testing and analysis in the project.
└── README.md                    | What you're reading right now!
```
### Process
##### Data Handling
The [dataset](https://www.kaggle.com/lantian773030/pokemonclassification) consists of 7,000 images of the 151 Pokémon from generation 1. Loading, resizing, and augmenting the data is done in what is essentially a single step using the ImageDataGenerator class from Keras. Using the flow_from_directory method, like in the example below, we can sample images from a directory, so long as the images are in seperate folder according to their class, automatically augment and resize them, and then pass them to the fit method all at once. 

This method also allows us to save on memory by taking advantage of the fit methods ability to train on generators. The downside of using this method is that each folder needs to have the same amount of images in it in order to avoid under or oversampling images from a certain class. This could be remedied in future using the class_weights parameter of each model's fit method. All the models are using the same pre-generated validation data.
```
# Instantiate ImageDataGenerator.
gen = ImageDataGenerator()
# Set up flow from training directory.
flow = gen.flow_from_directory('PokemonData/Train', target_size = (244, 244), batch_size = 20)
# Pass flow as data to a model's fit method.
model.fit(flow, epochs = 30)
```
Four different models were tested on the data, each trained for 30 epochs. In all there was one scratch built model, and three transfer learning models made from VGG16, Xception, and ResNet50 respectively. Each model was loaded with its imagenet weights, and the convolutional layers of each model were made untrainable in order to preserve those weights and avoid catastrophic forgetting. The tops of the models were removed and replaced with a dense layer and an output layer which accomodates the data. The structures of each model are listed below:

##### Model Structures
<details>
  <summary>Scratch Built Model</summary>
  
  ```
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  conv2d (Conv2D)              (None, 242, 242, 32)      896       
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 121, 121, 32)      0         
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 119, 119, 64)      18496     
  _________________________________________________________________
  max_pooling2d_1 (MaxPooling2 (None, 59, 59, 64)        0         
  _________________________________________________________________
  conv2d_2 (Conv2D)            (None, 57, 57, 64)        36928     
  _________________________________________________________________
  flatten (Flatten)            (None, 207936)            0         
  _________________________________________________________________
  dense (Dense)                (None, 64)                13307968  
  _________________________________________________________________
  dense_1 (Dense)              (None, 150)               9750      
  =================================================================
  Total params: 13,374,038
  Trainable params: 13,374,038
  Non-trainable params: 0
  _________________________________________________________________
  ```
</details>
<details>
  <summary>VGG16 (Transfer Learning)</summary>
  
  ```
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  vgg16 (Functional)           (None, 7, 7, 512)         14714688  
  _________________________________________________________________
  flatten_1 (Flatten)          (None, 25088)             0         
  _________________________________________________________________
  dense_2 (Dense)              (None, 128)               3211392   
  _________________________________________________________________
  dense_3 (Dense)              (None, 150)               19350     
  =================================================================
  Total params: 17,945,430
  Trainable params: 3,230,742
  Non-trainable params: 14,714,688
  _________________________________________________________________
  ```
</details>
<details>
  <summary>Xception (Transfer Learning)</summary>
  
  ```
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  xception (Functional)        (None, 8, 8, 2048)        20861480  
  _________________________________________________________________
  flatten_2 (Flatten)          (None, 131072)            0         
  _________________________________________________________________
  dense_4 (Dense)              (None, 128)               16777344  
  _________________________________________________________________
  dense_5 (Dense)              (None, 150)               19350     
  =================================================================
  Total params: 37,658,174
  Trainable params: 16,796,694
  Non-trainable params: 20,861,480
  _________________________________________________________________
  ```
</details>
<details>
  <summary>ResNet50 (Transfer Learning)</summary>
  
  ```
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  resnet50 (Functional)        (None, 8, 8, 2048)        23587712  
  _________________________________________________________________
  flatten_3 (Flatten)          (None, 131072)            0         
  _________________________________________________________________
  dense_6 (Dense)              (None, 128)               16777344  
  _________________________________________________________________
  dense_7 (Dense)              (None, 150)               19350     
  =================================================================
  Total params: 40,384,406
  Trainable params: 16,796,694
  Non-trainable params: 23,587,712
  _________________________________________________________________
  ```
</details>

### Results

### Sources
- [7,000 Labeled Pokemon](https://www.kaggle.com/lantian773030/pokemonclassification): The dataset used for this project.

### Dependencies
- [Python 3.9.7](https://www.python.org/)
- [NumPy 1.19.5](https://pypi.org/project/numpy/): Package for array computing with Python.
- [Pandas 1.3.3](https://pypi.org/project/pandas/): Python package that provides fast, flexible, and expressive data structures.
- [TensorFlow 2.6.0](https://pypi.org/project/tensorflow/): TensorFlow is an open source machine learning framework for everyone.
- [scikit-learn 0.24.2](https://pypi.org/project/scikit-learn/): A set of python modules for machine learning and data mining.
- [matplotlib 3.4.2](https://pypi.org/project/matplotlib/): Comprehensive library for creating static, animated, and interactive visualizations in Python.
