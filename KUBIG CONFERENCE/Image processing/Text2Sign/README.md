# 2021년 재난 안전 정보 수어 영상 데이터 구축 사업 위해 데모 번역 모델 (Python 3.11 버전 업 수행)
# Demo Korean to KSL Translation Model for 2021 NIA Emergency Alert Data Generation Project


------------------------------------------------------------------------------------------------------------------------
### Data Preprocessing

Before preprocessing the data, it should be split into three subsets: train, validation, and test.

The source data path should be specified with the "group_data_path" field in the config.json file.

We suggest using the given data split file (data_split.json) to reproduce our results, but a new file can be generated 
with the below script.
```shell
python preproc_createsplitref.py
```

NOTE: When files are copied with the below preprocessing methods, existing .json files in the target data subdirectories 
will be deleted. If you are running the preprocessing scripts for the first time, this will not be an issue. However, 
.json files manually placed in the data/train/, data/validate/, or data/test/ folders will be deleted.

##### Recommended script:
All preprocessing can be performed by running the following:
```shell
python preprocess.py
```

This script performs the following four steps:
- copies data from the source data directory to train/, test/, and validate/ subdirectories;
- attempts to move left and right hand signs that match but are misaligned by less than 0.1 seconds to the both sign 
  channel;
- generates a ksl dictionary based on data in train/; and
- preprocesses each data subdirectory and pickles the results in train_dataset.pkl, validate_dataset.pkl, 
  bleu_validate_dataset.pkl, test_dataset.pkl, and bleu_test_dataset.pkl files.
  

##### Alternative:
Alternatively, each preprocessing step can be performed separately.

```shell
python preproc_datasplit.py
python preproc_realign_annotations.py
python preproc_ksl_vocab.py
python preproc_pickle.py
```

------------------------------------------------------------------------------------------------------------------------
### Training

To train a model, set training parameters in config.json and run:
```shell
python train.py
```

Training and model parameters can be specified in the config.json (each parameter is explained in the Configuration
File section)

The translation model uses a standard transformer decoder pretrained on a Korean dataset (specifically, SKT's KoGPT2
public checkpoint is used) as a model backbone. This backbone can be finetuned with the other model parameters or
frozen, as specified in the config.json file.

------------------------------------------------------------------------------------------------------------------------
### Testing

##### BLEU
To test a model's BLEU score on the test data subset, set testing parameters in config.json and run:
```shell
python test.py
```
~~Testing may be slow since translations are first generated for the untrained model as a comparative baseline, and 
these translations are usually max length.~~
BLEU score for the untrained model does not need to be calculated as the probability of a BLEU score greater than zero
is effectively zero.

Note that the algorithm used to calculate BLEU score is the standard [NLTK Corpus BLEU](https://www.nltk.org/_modules/nltk/translate/bleu_score.html#corpus_bleu)
implementation.

##### Translation
All generated translations output two files: fileID_nia.json and fileID_player.json where fileID is dependent on the 
input method. The fileID_nia.json file matches the sign and NMS script structure of the training data. The 
fileID_player.json file is formatted for the avatar player. This player file should be used with the avatar player.

To generate translations of an input sentence, set appropriate testing parameters in config.json and run:
```shell
python translate_sentence.py "INPUT SENTENCE"
```
The translation results will be saved as two files named translate_sentence_TIMESTAMP_player.json and 
translate_sentence_TIMESTAMP_player.json at the location specified by the results_path field in the config.json file.

To generate translations from training data (in the project .json format), run:
```shell
python translate_file.py PATH
```
Where PATH specifies a path to a directory of files or to a single file. If the path points to a directory, each .json 
in the directory will be loaded and translated. Translations will be saved in a subdirectory labeled 
translate_file_TIMESTAMP/ at the location specified by the results_path field in the config.json file.

TIMESTAMP is the time the script was initialized in YYYYMMDDhhmmss format.

------------------------------------------------------------------------------------------------------------------------
### Configuration File

The config.json file stores all training and test parameters. A brief summary of each parameter can be seen below.

```json5
{
  "group_data_path": "data/groups/", // Path to data (all files at or in subfolders should be relevant
                                     //.json data files)

  "data_split_path": "data/data_split.json", // Path to data split (train, validate, test) file
  "ksl_vocab_path": "data/ksl_vocab.json", // Path to ksl gloss vocab file (extracted from the training
                                           // set only)

  "animation_mapping_path": "data/animation_mapping.json", // Path to animation mapping file (for
                                                           // player formatting) 

  "split_group_by_base": false, // Flag to group instances by base id when calculating the data split

  "use_dataset_pickles": true, // Load from pickled datasets (faster, but requires running 
                               // Prepare_data.py or preprocess.py)
  "pickle_directory": "data/", // Location of pickled dataset
  "train_pickle_name": "train_dataset.pkl", // Train data pickle 
  "val_pickle_name": "validate_dataset.pkl", // Validation data pickle
  "bleu_val_pickle_name": "bleu_validate_dataset.pkl", // Validation BLEU data pickle
  "test_pickle_name": "test_dataset.pkl", // Test data pickle
  "bleu_test_pickle_name": "bleu_test_dataset.pkl", // Test BLEU data pickle

  "data_train_path": "data/train/", // Path to train data subset
  "data_validate_path": "data/validate/", // Path to validation data subset
  "data_test_path": "data/test/", // Path to test data subset

  "preload_data": true, // Flag to preload data (set to false if available memory limited)

  "max_source_length": 256, // Max source sequence length (post-tokenization)
  "max_target_length": 256, // Max target sequence length

  "channel_to_gloss": false, // Flag to use channel value to predict gloss value

  "ksl_embedding_method": "concat", // Flag to specify method of embedding KSL glosses
  "ksl_embedding_split": 0.9, // Split ratio for embedding KSL glosses. If set as 0<a<1, then
                              // a * embedding size of the KSL embedding is based on the KSL gloss and
                              // (1-a) * embedding size comes from the gloss channel

  "ignore_nms": true, // Flag to not train on NMS features (used for BLEU score benchmarking)

  "train": { // Sub configuration variables for the training script
    "device": "cuda:0", // If training on the CPU, use 'cpu' otherwise, use 'cuda:n' where n is the gpu 
                        // number
    "n_dl_workers": 0, // Number of data loader workers (set to main process using 0 until bug is fixed)
    "val_n_dl_workers": 0, // Number of validation data loader workers
    "batch_size":8, // Training batch size
    "val_batch_size": 8, // Validation batch size
    "accumulate": 10, // How often to perform gradient accumulation
    "freeze_strategies": [ // Which freezing strategy to use (selected by the freeze_pointer flag below)
      "none",
      "pretrained",
      "attention"
    ],
    "freeze_pointer": 0, // Flag to select one of the above freezing strategies
    "unfreeze_step": 2000, // Step to train to before implementing the above freezing strategy (only the
                           // Embedding and output layers will be trained until this step) 
    "print_every": 1, // Log frequency (while training)
    "eval_every": 2, // Evaluation frequency (while training)
    "bleu_every": 5, // BLEU score calculation frequency (while training)
    "save_model_every": 5, // Model save frequency (while training)
    "epochs": 60, // Number epochs to train over
    "optimizers": [ // Which optimizer to use (selected by the optimizer_pointer flag below)
      "sgd",
      "adam",
      "adamw"

    ],
    "optimizer_pointer": 2, // Flag to select one of the above optimizers. Note that there is a PyTorch
                            // bug with AdamW in versions < 1.9
    "model_lr": 0.0001, // Learning rate for pretrained model backbone
    "point_lr": 0.001, // Learning rate for pointer module
    "sign_lr": 0.001, // Learning rate for sign and NMS head
    "timing_lr": 0.001, // Learning rate for timing predictio nhead
    "patience": 3, // Patience for LR scheduler (uses PyTorch's ReduceLROnPlateau scheduler)
    "ksl_embedding_init_pretrained": false, // Flag to align special tokens with pretrained special
                                            // tokens
    "load_model": false, // Flag to load custom model weights
    "load_optimizer": false, // Flag to load custom optimizer parameters
    "load_id": "ignore_nms", // Load ID for custom model and optimizer. Model and optimizer should be named
                             // [load_id]_model.pt and [load_id]_optimizer.pt, respectively (where [load_id]
                             // is replaced with the value specified here).
    "model_path": "checkpoints/", // Path to model checkpoint directory
    "experiment_id": "ignore_nms", // Save ID for the current training run
    "n_beams": 2, // Number of beams to use when calculating the validation BLEU score (note that the beam
                  // search algorithm used is not optimized and will be extremely slow--it is recommended to
                  // use only a few beams during training validation)
    "top_k": null, // K for top K sampling. Note that sampling cannot be used with beam search.
    "top_p": null // P value for nucleus sampling. Note that sampling cannot be used with beam search.
  },

  "test": { // sub configuration variables for test and inference scripts
    "device": "cuda:1", // If testing on the CPU, use 'cpu' otherwise, use 'cuda:n' where n is the gpu
                        // number
    "n_dl_workers": 0, // Number of data loader workers (set to main process using 0 until bug is fixed)
    "batch_size": 8, // Testing batch size
    "print_every": 1, // Log frequency (while testing)
    "eval_every": 1, // Evaluation frequency (while testing)
    "load_model": true, // Flag to load custom model weights
    "load_id": "ignore_nms_ep20", // Load ID for custom model and optimizer. Model and optimizer should be
                                  // named [load_id]_model.pt and [load_id]_optimizer.pt, respectively
                                  // (where [load_id] is replaced with the value specified here).
    "model_path": "checkpoints/", // Path to model checkpoint directory
    "results_path": "results/", // path to results directory (where translations are saved)
    "n_beams": 5, // Number of beams to use when calculating the test BLEU score 
    "top_k": null, // K for top K sampling. Note that sampling cannot be used with beam search.
    "top_p": null // P value for nucleus sampling. Note that sampling cannot be used with beam search.
  }
}
```
