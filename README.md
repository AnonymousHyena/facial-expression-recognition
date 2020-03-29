# facial-expression-recognition

# Abstract
Humans use, mainly, their facial expressions to convey their emotional state. Automatic expression recognition though, still remains a very interesting and at the same time challenging problem to solve, since the way and the magnitude at which somebody changes their facial expression differs in many ways from person to person. On the other hand, the need for a reliable expression recognition is critical for human-computer interaction applications, emotional computing and artificial intelligence in general.

For this thesis, a deep learning model was developed and trained in order to detect one of the 7 basic emotional states, namely happiness, sadness disgust, surprise, anger, fear, and the neutral state. The training and evaluation of the model were conducted using photographs from the MUG Facial Expression database<sup>1</sup>, of the Multimedia Understanding Group of the Aristotle University of Thessaloniki.

In greater detail, the model that was developed was trained in two phases. In the first, the convolutional layers of the model were trained like an autoencoder, with this phase serving as the initialisation step of our model, while in the second, the whole model was trained to recognise the facial expressions. Finally, to improve the model’s ability to differentiate between emotions, the initial dataset was augmented by adding new photos that were created by applying a number of transformations on our data, like for example a change in brightness, zoom or rotation of the image.

From the evaluation of our model, we can see that our method produces results that can closely challenge the state of the art while keeping the size of our model to a minimum. From our experiments, we can conclude that the data augmentation provided a significant improvement in our model’s performance, while our initialisation technique did not offer any notable betterment to our model.

<sup>1</sup> N. Aifanti, C. Papachristou, and A. Delopoulos. The mug facial expression database. In Proc. 11th Int. Workshop on Image Analysis for Multimedia Interactive Services (WIANIS), Desenzano, Italy, Arpil 2010

# Setup
To set up the enviroment and start the training proccess run `setup.sh`

## Restart
To trash everything and start over run `restart.sh`