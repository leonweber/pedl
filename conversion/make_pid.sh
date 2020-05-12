#!/usr/bin/env bash

# Raw to JSON
##
python conversion/raw_ds_to_json_format.py --raw data/PathwayCommons11.pid.hgnc.txt_raw/*train* --data data/PathwayCommons11.pid.hgnc.txt.train.json --out distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.json
python conversion/raw_ds_to_json_format.py --raw data/PathwayCommons11.pid.hgnc.txt_raw/*dev* --data data/PathwayCommons11.pid.hgnc.txt.dev.json --out distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.json
python conversion/raw_ds_to_json_format.py --raw data/PathwayCommons11.pid.hgnc.txt_raw/*test* --data data/PathwayCommons11.pid.hgnc.txt.test.json --out distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.json

## Mask entities
###
#python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.json
#python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json
#python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json
#
#
## Generate PEDL data
###
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#
#
## Generate comb-dist data
###
#python -m conversion.ds_to_comb_dist_relex distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.json distant_supervision/comb_dist_direct_relex/data/PathwayCommons11.pid.hgnc.txt/train_masked.json --direct_data distant_supervision/data/BioNLP-STs/all_masked.json --pair_blacklist distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json
#python -m conversion.ds_to_comb_dist_relex distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json distant_supervision/comb_dist_direct_relex/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json
#python -m conversion.ds_to_comb_dist_relex distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json distant_supervision/comb_dist_direct_relex/data/PathwayCommons11.pid.hgnc.txt/test_masked.json
#
#
## Generate 2012 data
###
##python conversion/ds_filter_by_year.py distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.json --max_year 2012
##python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
##python -m conversion.ds_to_comb_dist_relex distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.json distant_supervision/comb_dist_direct_relex/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.json
