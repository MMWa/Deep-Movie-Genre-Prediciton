# Deep Movie Genre Prediction
Give the title and description of a movie can the genre be predicted?
## About
This repo contains the code for training and running a model the would predict a movies genre given the title and a description.

## How to run
A prediction can be done by calling test.py with the title and description of a movie.

For example:
```
test.py --title "Othello" --description "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic."
```
The returned prediction is printed to console, in the following format.
```
{
"title": "Othello",
"description": "The evil Iago pretends to be friend of Othello in
order to manipulate him to serve his own end in the film version of
this Shakespeare classic.",
"genre": "Drama"
}
```
## Requirements
To run this solution the following libraries are needed.
* Numpy
* Keras
* Tensorflow
* SciPy

## Potential Improvements
* The number of genres can be reduced.
* The samples per class can be better balanced to reduce model bias.
