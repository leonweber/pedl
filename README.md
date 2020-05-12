# PEDL

PEDL is a method for predicting protein-protein assocations from text. The paper describing it will be presented at ISMB 2020.

## Requirements
* `python >= 3.6`
* `pip install -r requirements.txt`
* `pytorch >= 1.3.1` (has to be installed manually, due to different CUDA versions)

## Generate data
We use two types of data sets: Data generated from the BioNLP-ST event extraction data sets and the distantly supervised PID data set

### Generate BioNLP
`./conversion/make_bionlp_data.sh` generates the BioNLP data sets for both PEDL and [comb-dist](https://github.com/allenai/comb_dist_direct_relex/tree/master/relex)

All experiments in the paper have been performed with the masked version of the data, e.g. `distant_supervision/data/BioNLP-ST_2011/train_masked.json`.

### Generate PID
Generating the PID data is a bit more involved:

1. First, we have to download the raw PubMed Central texts: `python download_pmc.py`. CAUTION: This produces over 200 GB of files and spawns multiple processes.
2. Then, we have to download the [PubTator Central file](ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.gz) and place it into the root directory. This file consumes another 80 GB when decompressed.
3. Generate the raw PID data: `./conversion/generate_raw_pid.sh`
4. Generate the final PID data: `./conversion_make_pid.sh`


## Training PEDL
Before training, [SciBERT](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar) has to be downloaded and placed to some directory (called `$bert_dir` from now on). 

The vocabulary of SciBERT has to be adapted to include the entity markers and protein masks: `cp distant_supervision/vocab.txt $bert_dir`

PEDL can be trained with `python -m distant_supervision.train_pedl`, (see `train_pedl.sh` for exact suitable arguments.

If you just want to reproduce the experiments from the paper, this can be achieved with `./train_pedl.sh`.

## Pretrained model
As an alternative to training your own model, you can use [this version of PEDL](https://drive.google.com/open?id=1Toh49LDPdB8SoyRnhoO43HBC_nG4Ur3I) that was trained on PID and used for the experiments in the paper.

## Predicting with PEDL
The trained PEDL model can be used to predict PPAs for a new data set. See `predict_pedl.sh` for details.



## Disclaimer
Note, that this is highly experimental research code which is not suitable for production usage. We do not provide warranty of any kind. Use at your own risk.

