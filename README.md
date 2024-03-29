# Deep Movie Genre Prediction
Give the title and description of a movie can the genre be predicted?
## About
This repo contains the code for training and running a model the would predict a movie's genre given the title and a description.
### The model
The model used is a deep model, containing two input heads each with their own embedding layers and LSTM layers.

The output from each LSTM layer is concatenated then the result is fed into two dense layers where an output classification is given. 
## How to run
A prediction can be done by calling test.py with the title and description of a movie as arguments.

For example:
```
test.py --title "Othello" --description "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic."
```
The returned prediction is printed to console, in the following format.
```
{
'title': 'Othello',
'description': 'The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.', 
'genre': 'Action'
}
```
## How to train
To train the classifier, the train.ipynb can be used, simply replace the filename and make sure the correct columns are selected, the notebook should handle the rest.
## Requirements
To run this solution the following libraries are needed.
* Numpy
* Keras
* Tensorflow
* SciPy
## Potential Improvements
* The number of genres can be reduced.
* The samples per class can be better balanced to reduce model bias.
