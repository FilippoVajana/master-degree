# Valutazione di metriche di incertezza per reti neurali

Obiettivo di questo lavoro è sviluppare metodologie per monitorare il grado di incertezza insito negli output di una rete neurale.

# Basic usage:

## install depedencies

    $ pip install -r requirements.txt (?)

## Model training

    $ python train_model.py 
    
usa le impostazioni in runconfig.json per addestrare da zero la LeNet5 su MNIST.


## Model tesing

    $ python test_model.py

testa il modello e tira fuori le metriche.
