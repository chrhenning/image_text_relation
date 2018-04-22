# Comparison to conventional machine learning tools

This folder provides a script that trains a SVM with extracted article embeddings as features

The *autoencoder features* are article embeddings generated from an encoder network trained by the autoencoder (and not further modified by the classifier).

The *classifier features* are article embeddings generated from an encoder network pretrained by the autoencoder and fintuned by the classifier.
