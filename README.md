# A Singing Voice Conversion model using the [AutoVC framework](https://github.com/auspicious3000/autovc), with an improved loss function.

This repository reflects the research presented in our paper A COMPARATIVE ANALYSIS OF LATENT REGRESSOR LOSSES FOR SINGING VOICE CONVERSION, as submitted to Sound and Music Computing Conference 2023. It is a compilation of the following submodules which facilitate feature generation, training a voice identity encoder, :
* [singer-identity-encoder](https://github.com/Trebolium/singer-identity-encoder)
* [autoSvc](https://github.com/Trebolium/autoSvc)
* [my_utils](https://github.com/Trebolium/my_utils)

This repository comes with a small set of audio files for training and validation. These are found in the ```singer-identity-encoder/example_audio``` directory. To train the network, you will need to replace it with a more substantial dataset. The training-validation cycles and number of iterations are by default set very low, intending to demonstrating proof of concept quickly. These hyperparameters, along with many others and pathways are configurable although the interface used to change these is different between the singer-identity-encoder and autoSvc repositories (see original documentation for more details on this).

## Generate audio features

To convert the audio files to melspectrogram features, run ```python singer-identity-encoder/audio_to_features.py```. This will automatically save the feature numpy files to ```singer-identity-encoder/example_feats```.

## Pretrain SIE encoder

To train the singer identity encoder using these generated features, run ```python singer-identity-encoder/main.py```. This will train an Singer Identity Embedding (SIE) model and automatically save it to ```singer-identity-encoder/sie_models```, populated with input feature examples, configuration text file, a ```saved_model.pt``` file created when validation loss improves, and a ```ckpt_#.ckpt``` file created when training is finished.

## Generate SIE lookup table

Now that the SIE model is trained, we can generate an average SIE for each singer across all of their recordings in the given dataset. To do this, run ```python singer-identity-encoder/generate_sie_table.py```, which saves the resulting SIEs to a directory called ```voice_embs_visuals_metadata```.

## Train autoSvc

