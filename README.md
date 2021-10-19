### Motivation
The goal of this project is to create a classifier that can identify 1st generation pokemon. Despite what the name suggests, this is not a classifier designed to play the "Who's that pokemon?" game from the anime, though images from that game will be tested later. As far as what the motivation behind this project is, I'm curious how a CNN will be able to adapt to images that are mainly hand drawn (specifically whether transfer learning will be less effective in this context) and how good performance will be with 151 different classes.

### File Structure
```
.
├── PokemonData                  | All the images used to train and validate every model.
│   ├── Test
│       ├── Abra
│       │   ├── .jpg
│       │   ├── .jpg
│       │   ├── ...
│       ├── Aerodactyl
│       ├── Alakazam
│       ├── ...
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
