# Singing Voice Converter
## Uses an enhanced version of the [AutoVC framework](https://github.com/auspicious3000/autovc) with many extra features, and an improved loss function.

This repository reflects the research presented in our paper [A COMPARATIVE ANALYSIS OF LATENT REGRESSOR LOSSES FOR SINGING VOICE CONVERSION](https://arxiv.org/abs/2302.13678), as submitted to Sound and Music Computing Conference 2023. It is a compilation of the following submodules which facilitate feature generation, training a voice identity encoder, :
* [singer-identity-encoder](https://github.com/Trebolium/singer-identity-encoder)
* [autoSvc](https://github.com/Trebolium/autoSvc)
* [my_utils](https://github.com/Trebolium/my_utils)

This repository comes with a small set of audio files for training and validation. These are found in the ```singer-identity-encoder/damp_example_audio``` directory. To train the network, you will need to replace it with a more substantial dataset. The training-validation cycles and number of iterations are by default set very low, intending to demonstrating proof of concept quickly. These hyperparameters, along with many others and pathways are configurable although the interface used to change these is different between the singer-identity-encoder and autoSvc repositories (see original documentation for more details on this).

When using this framework to train your own models, please take care to ensure you examine the argparse argument options, or the parameter files. The current default values are set to make the repository work quickly.

## Initialisation

Before attempting to run any python code, ensure that you have created a virtual environment within which ```pip install -r requirements.txt``` can be run. Then initialise the submodules by running ```git submodule update --init```.

## Singer Identity Encoder Model

### Generate audio features

To convert the audio files to melspectrogram features, run ```python singer-identity-encoder/audio_to_features.py```. This will automatically save the feature numpy files to ```singer-identity-encoder/damp_example_feats```.

### Pretrain SIE encoder

To train the singer identity encoder using these generated features, run ```python singer-identity-encoder/main.py```. This will train an Singer Identity Embedding (SIE) model and automatically save it to ```singer-identity-encoder/sie_models```, populated with input feature examples, configuration text file, a ```saved_model.pt``` file created when validation loss improves, and a ```ckpt_#.ckpt``` file created when training is finished. To produce a well trained network, users are encouraged to provide the path to a realistic dataset and adjust the validation and training iterations to a substantial size using the appropriate flags. For example: ```python singer-identity-encoder/main.py -fd path/to/dataset -ti 1000 -vi 100```.

### Generate SIE lookup table

Now that the SIE model is trained, we can generate an average SIE for each singer across all of their recordings in the given dataset. To do this, run ```python singer-identity-encoder/generate_sie_table.py```, which saves the resulting SIEs to a directory at ```./voice_embs_visuals_metadata/default_model/damp_example_feats``` (assuming variables remain at default settings). Unlike ```singer-identity-encoder/main.py```, the ```generate_sie_table.py``` script uses the parameter file ```avg_emb_params.py``` for its arguments.

### Plot your embeddings on a 2D plane

Plot the embeddings of each singer you have generated SIEs for on a 2D plane to verify salience between singers, implying how robust the SIEs descriminative capabilities are. To do this, run ```python singer-identity-encoder/plot_embs_tsne.py```. Output visualisations are sent to their default location at ```./voice_embs_visuals_metadata/default_model/damp_example_feats/val```. Users can use this script's argparse flags to set the path for the directory containing the pickled SIE data. This script is made to work specifically with the Vocadito, VCTK, or DAMP dataset, and requires the relevant csv files relating to gender (a csv file containing information for the supplied example dataset is provided).

## AutoSVC model

This particular version of AutoVC has been adapted from the work documented in [(Qian et. al, 2019)](https://proceedings.mlr.press/v97/qian19c/qian19c.pdf), thanks to their supplied respository, [AutoVC](https://github.com/auspicious3000/autovc).

### Training phase

Using the SIE lookup tables generated from the previous step, we can now train AutoSVC, which is designed to take a source and target singer during test time, and superimpose the identity features of the target singer onto the source singer, thereby achieving singing voice identity conversion.

### Training phase

Running ```python autoSvc/main.py``` will train the AutoSvc network.

You can either keep the default paths variables unchanged, use a pretrained SIE model (by downloading it to the ```singer-identity-encoder/sie_models``` directory, or training it yuorself). If use a pretrained model, change the varialbe ```SIE_model_path``` in the ```train_params.py``` file to point towards the directories you require. Please note that the values for size of training iterations (```max_cycle_iters```) and max iterations (```max_iters```) are extremely small, and will likely need to be adjusted for any training towards useful inference.  

### Conversion phase

You will need to download the WaveNet model, available at the original [repository](https://github.com/auspicious3000/autovc), and add it to the ```autoSvc``` directory. If this does not work, please contact me for a copy.

Simply running ```python autoSvc/convert_synthesize/pitchmatched_conversion.py``` will by default search for the model manually trained in the previous step. If you wish to use the pretrained model or train your own model, download it or train it, and make sure it is saved in the ```autoSvc/models``` directory, and use the flag ```--model_name``` to specify which model is to be used for voice conversion. You must also make sure that lookup tables generated by this pretrained model are available in the ```voice_embs_visuals_metadata``` directory in the parent directory containing this repository.

After running each command, voice converted audio files are generated and saved in ```autoSvc/converted_audio```.