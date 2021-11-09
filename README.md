### Summary
The goal of this project is to make a image classifier which can identify any of the first generation pokémon. The idea behind this is that once a sufficiently accurate classifier has been made, it can be used to build software similar to a pokédex, which is a device that can identify and provide information about any pokémon it is shown.

To achieve this goal, I first loaded, resized, and randomly augmented the training data to make a total of 5000 training images and 2000 validation images. Those images were then used to train two models, one with a scratch made structure and randomized weights, and one made using VGG16 pretrained with ImageNet weights. I then trained both for 30 epochs. The results were that the transfer learning model performed much better, attaining a validation accuracy of ~73%, compared to the scratch built model's validation accuracy of ~38%.

If you would prefer the article form of this project it can be found [here!](https://linkwentz.medium.com/whos-that-pok%C3%A9mon-building-a-pok%C3%A9mon-identifier-in-keras-e8e1078f1b14)

### Directory
```
.
├── Plots                        | Every visualization generated in the notebook.
│   ├── .jpg
│   ├── .jpg
│   └── ...
├── PokémonData                  | All the images used to train and validate every model.
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
├── pokémon_classification.ipynb | All model testing and analysis in the project.
└── README.md                    | What you're reading right now!
```

### Sources
- [7,000 Labeled Pokémon](https://www.kaggle.com/lantian773030/pokemonclassification): The dataset used for this project.
- [Pokémon Database](https://pokemondb.net/pokedex/national): The source for all the sprites used for "Who's That Pokémon"

### Dependencies
- [Python 3.9.7](https://www.python.org/)
- [Pillow 8.3.1](https://pypi.org/project/Pillow/): Pillow is the friendly PIL fork by Alex Clark and Contributors.
- [NumPy 1.19.5](https://pypi.org/project/numpy/): Package for array computing with Python.
- [Pandas 1.3.3](https://pypi.org/project/pandas/): Python package that provides fast, flexible, and expressive data structures.
- [TensorFlow 2.6.0](https://pypi.org/project/tensorflow/): TensorFlow is an open source machine learning framework for everyone.
- [scikit-learn 0.24.2](https://pypi.org/project/scikit-learn/): A set of python modules for machine learning and data mining.
- [matplotlib 3.4.2](https://pypi.org/project/matplotlib/): Comprehensive library for creating static, animated, and interactive visualizations in Python.
