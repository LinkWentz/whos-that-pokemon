### Motivation
The goal of this project is to create a classifier that can identify 1st generation pokemon. Despite what the name suggests, this is not a classifier designed to play the "Who's that pokemon?" game from the anime, though images from that game will be tested later. The motivation behind this project is mainly to get more experience with image recognition by testing different approaches on the same dataset; though depending on the results, I am considering some different applications for these models. Specifically though we are trying to determine whether transfer learning is better for this application.
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
The [dataset](https://www.kaggle.com/lantian773030/pokemonclassification) consists of 7,000 images of the 149 Pokémon from generation 1. Loading, resizing, and augmenting the data is done in what is essentially a single step using the ImageDataGenerator class from Keras. Using the flow_from_directory method, like in the example below, we can sample images from a directory - so long as the images are in seperate folders according to their classes - and automatically augment and resize them and resize them all at once.

```
```

This is where we encounter the main difficulty I had during this project: hardware limitations. I'll spare you the details, but it came down to two options for how the data would be passed to the fit methods of the models. The first was to pass the flow of the ImageDataGenerator directly to the models. This generates data in batches for each epoch. The advantages of this method are three-fold. It allows you to conserve memory by only having one batch of data in memory at a time, while also ensuring that your models never train on the exact same image twice, which prevents overfitting and exposes your model to much more variance. However the drawback of this method is that the CPU is used quite  a lot more, and as I was using local resources and my CPU is almost 10 years old at this point, this method added 30 seconds to each epoch.

This led to the second method, which was to use the ImageDataGenerator to generate a dataset of an arbitrary size and use that to train the models. This method has the exact opposite problems. All the images are held in memory at once, which means you can only have so complex of a model. Additionally, as the images are the same for each epoch, the model can overfit much more easily. However, epochs take much less time with this method, making it much more time efficient. Ultimately this is the method I went with as it made fine tuning the models much easier.

##### Model Design
Three different models were tested on the data, each trained for 30 epochs. In all there was one scratch built model, and two transfer learning models made from VGG16 and Xception. For the scratch model, I designed it by simply starting with a basic model, and making it more and more complex by adding more layers, filters, and neurons. I'm sure I could have improved it even more but I am happy with where I was able to get performance. For the transfer learning models, each model was loaded with its imagenet weights from the keras applciations module, and the convolutional layers of each model were made untrainable in order to preserve those weights and avoid catastrophic forgetting. The tops of the models were removed and replaced with a dense layer and an output layer which accomodates the data. 

I tested several learning rates, though for the sake of simplicity I used the same optimizer for each model. I found that a learning rate of 0.0001 worked well for each model, though I did introduce one of the two callbacks I would end up using just in case. Callbacks in keras allow you to do a number of useful things during training. In this case, I used the ReduceLROnPlateau callback, which allows to reduce the learning rate when you don't see improvement in a metric after a certain amount of epochs. Sometimes you can "circle the drain" in training, usually because your learning rate is too high, so this aims to help with that and squeeze out some extra performance during training. The second callback I used was the ModelCheckpoint callback, which I used to save the model whenever it reached a new peak validation accuracy. This meant that after training, even if the model overfit, the model that was saved would be the best model found during training.

The structures of each model are listed below:

##### Model Structures
<details>
  <summary>Scratch Built Model</summary>
  
  ```
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  conv2d (Conv2D)              (None, 222, 222, 16)      448       
  _________________________________________________________________
  dropout (Dropout)            (None, 222, 222, 16)      0         
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 111, 111, 16)      0         
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 109, 109, 32)      4640      
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 109, 109, 32)      0         
  _________________________________________________________________
  max_pooling2d_1 (MaxPooling2 (None, 54, 54, 32)        0         
  _________________________________________________________________
  conv2d_2 (Conv2D)            (None, 52, 52, 64)        18496     
  _________________________________________________________________
  dropout_2 (Dropout)          (None, 52, 52, 64)        0         
  _________________________________________________________________
  max_pooling2d_2 (MaxPooling2 (None, 26, 26, 64)        0         
  _________________________________________________________________
  conv2d_3 (Conv2D)            (None, 24, 24, 64)        36928     
  _________________________________________________________________
  dropout_3 (Dropout)          (None, 24, 24, 64)        0         
  _________________________________________________________________
  max_pooling2d_3 (MaxPooling2 (None, 12, 12, 64)        0         
  _________________________________________________________________
  conv2d_4 (Conv2D)            (None, 10, 10, 64)        36928     
  _________________________________________________________________
  dropout_4 (Dropout)          (None, 10, 10, 64)        0         
  _________________________________________________________________
  flatten (Flatten)            (None, 6400)              0         
  _________________________________________________________________
  dense (Dense)                (None, 256)               1638656   
  _________________________________________________________________
  dropout_5 (Dropout)          (None, 256)               0         
  _________________________________________________________________
  dense_1 (Dense)              (None, 256)               65792     
  _________________________________________________________________
  dropout_6 (Dropout)          (None, 256)               0         
  _________________________________________________________________
  dense_2 (Dense)              (None, 149)               38293     
  =================================================================
  Total params: 1,840,181
  Trainable params: 1,840,181
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
  dense_3 (Dense)              (None, 512)               12845568  
  _________________________________________________________________
  dropout_7 (Dropout)          (None, 512)               0         
  _________________________________________________________________
  dense_4 (Dense)              (None, 149)               76437     
  =================================================================
  Total params: 27,636,693
  Trainable params: 12,922,005
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
  xception (Functional)        (None, 7, 7, 2048)        20861480  
  _________________________________________________________________
  flatten_2 (Flatten)          (None, 100352)            0         
  _________________________________________________________________
  dense_5 (Dense)              (None, 512)               51380736  
  _________________________________________________________________
  dropout_8 (Dropout)          (None, 512)               0         
  _________________________________________________________________
  dense_6 (Dense)              (None, 149)               76437     
  =================================================================
  Total params: 72,318,653
  Trainable params: 51,457,173
  Non-trainable params: 20,861,480
  _________________________________________________________________
  ```
</details>

### Results
After fine tuning, I was able to a fairly good place in terms of performance, at least in my opinion. The scratch model wound up at around 68% validation accuracy, which is not the best it could be, but something that, after a lot of tuning, I was pretty happy with. As for transfer learning, both models had a validation accuracy above 90%, which I was also fairly happy with. The full results for each model including plots of accuracy and loss over time are listed below:

<details>
  <summary>Scratch Built Model</summary>
  
  ```
  loss__________________________
       Max: 4.87523
       Min: 0.15865
  categorical_accuracy__________
       Max: 0.95100
       Min: 0.02067
  val_loss______________________
       Max: 4.64428
       Min: 1.29255
  val_categorical_accuracy______
       Max: 0.68200
       Min: 0.04300
  lr____________________________
       Max: 0.00010
       Min: 0.00010
  ```
  ![Accuracy](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/scratch_model_accuracy.jpg)
  ![Loss](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/scratch_model_loss.jpg)
  
</details>
<details>
  <summary>VGG16 (Transfer Learning)</summary>
  
  ```
  loss__________________________
       Max: 4.08060
       Min: 0.02365
  categorical_accuracy__________
       Max: 1.00000
       Min: 0.21717
  val_loss______________________
       Max: 2.79893
       Min: 0.35582
  val_categorical_accuracy______
       Max: 0.94000
       Min: 0.49600
  lr____________________________
       Max: 0.00010
       Min: 0.00000
  ```
  ![Accuracy](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/vgg16_model_accuracy.jpg)
  ![Loss](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/vgg16_model_loss.jpg)
</details>
<details>
  <summary>Xception (Transfer Learning)</summary>
  
  ```
  loss__________________________
       Max: 3.35028
       Min: 0.01567
  categorical_accuracy__________
       Max: 1.00000
       Min: 0.32433
  val_loss______________________
       Max: 1.82259
       Min: 0.45313
  val_categorical_accuracy______
       Max: 0.91100
       Min: 0.62100
  lr____________________________
       Max: 0.00010
       Min: 0.00010
  ```
  ![Accuracy](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/xception_model_accuracy.jpg)
  ![Loss](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/xception_model_loss.jpg)
</details>
As for how the models perform in the game itself, the models each made predictions on 9 silhouettes of the same 9 pokemon. Here are the silhouettes for reference and if you would like to play along:

![Silhouettes](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/Silhouettes.jpg)

And here are the answers and the predictions each model made:

![Silhouette Predictions](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/silhouette_predictions.jpg)

As you can see the scratch model performed the worst, seeming to favor magnemite the majority of the time. In all the scratch model actually failed to guess even a single pokemon correctly. VGG16 performed better, getting Dugtrio, though still showing that favoritism behaiviour towards Rapidash. Xception performed the best managing to guess both Articuno and Rhydon, though overall none of the models performed especially well. Obviously, the models weren't trained to handle this use case though. If we wanted to do that, we would use silhouettes as training data. So, let's look at how the models perform when predicting on the full color versions instead:

![Color Predictions](https://github.com/LinkWentz/whos-that-pokemon/blob/master/Plots/color_predictions.jpg)

Here we can see the model performance is much more consistent, though not perfect. The scratch model for its part still fails to identify a single pokemon, and still favors Magnemite. The transfer learning models however perform much better. VGG16 and Xception both correctly identified 6 pokemon, and in fact almost the same 6 save for Rhydon and Nidorina. They do fail to recognize Dodrio and Kingler however, which suggests that there is still many improvements to be made.

Overall, transfer learning seems to be a much better approach, or at least more efficient. Given time and more computing resources we could of course develop a much better scratch model, but as it stands, transfer learning is much easier to achieve good performance with.

### Sources
- [7,000 Labeled Pokemon](https://www.kaggle.com/lantian773030/pokemonclassification): The dataset used for this project.

### Dependencies
- [Python 3.9.7](https://www.python.org/)
- [NumPy 1.19.5](https://pypi.org/project/numpy/): Package for array computing with Python.
- [Pandas 1.3.3](https://pypi.org/project/pandas/): Python package that provides fast, flexible, and expressive data structures.
- [TensorFlow 2.6.0](https://pypi.org/project/tensorflow/): TensorFlow is an open source machine learning framework for everyone.
- [scikit-learn 0.24.2](https://pypi.org/project/scikit-learn/): A set of python modules for machine learning and data mining.
- [matplotlib 3.4.2](https://pypi.org/project/matplotlib/): Comprehensive library for creating static, animated, and interactive visualizations in Python.
