#!/usr/bin/env bash
# Only run from base dir!


## Convert to distant supervision format
####
cd data
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_train_data_rev1.tar.gz'
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_devel_data_rev1.tar.gz'
tar xf BioNLP-ST_2011_genia_train_data_rev1.tar.gz
tar xf BioNLP-ST_2011_genia_devel_data_rev1.tar.gz
cd ..

python conversion/standoff_to_ds.py --data data/BioNLP-ST_2011_genia_train_data_rev1/ --out distant_supervision/data/BioNLP-ST_2011_genia/train.json
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2011_genia_devel_data_rev1/ --out distant_supervision/data/BioNLP-ST_2011_genia/dev.json

cd data
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_training_data_rev1.tar.gz'
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_development_data_rev1.tar.gz'
tar xf BioNLP-ST_2011_Epi_and_PTM_training_data_rev1.tar.gz
tar xf BioNLP-ST_2011_Epi_and_PTM_development_data_rev1.tar.gz
cd ..
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2011_Epi_and_PTM_training_data_rev1/ --out distant_supervision/data/BioNLP-ST_2011_epi/train.json
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2011_Epi_and_PTM_development_data_rev1/ --out distant_supervision/data/BioNLP-ST_2011_epi/dev.json

cd data
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_train_data_rev3.tar.gz?attredirects=0'
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_devel_data_rev3.tar.gz?attredirects=0'
tar xf BioNLP-ST-2013_GE_train_data_rev3.tar.gz?attredirects=0
tar xf BioNLP-ST-2013_GE_devel_data_rev3.tar.gz?attredirects=0
cd ..
python conversion/standoff_to_ds.py --data data/BioNLP-ST-2013_GE_train_data_rev3/ --out distant_supervision/data/BioNLP-ST_2013_GE/train.json
python conversion/standoff_to_ds.py --data data/BioNLP-ST-2013_GE_devel_data_rev3/ --out distant_supervision/data/BioNLP-ST_2013_GE/dev.json

cd data
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_training_data.tar.gz?attredirects=0'
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_development_data.tar.gz?attredirects=0'
tar xf 'BioNLP-ST_2013_PC_training_data.tar.gz?attredirects=0'
tar xf 'BioNLP-ST_2013_PC_development_data.tar.gz?attredirects=0'
cd ..
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2013_PC_training_data/ --out distant_supervision/data/BioNLP-ST_2013_PC/train.json
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2013_PC_development_data/ --out distant_supervision/data/BioNLP-ST_2013_PC/dev.json


## Clean up
###

cd data
rm -f BioNLP*.tar.gz*


## Combine individual data sets
###

cd ..
python conversion/combine_bionlp_ds_data.py distant_supervision/data/BioNLP-ST*2013* distant_supervision/data/BioNLP-ST_2013
python conversion/combine_bionlp_ds_data.py distant_supervision/data/BioNLP-ST*2011* distant_supervision/data/BioNLP-ST_2011


## Mask entities
###

python -m conversion.ds_tag_entities distant_supervision/data/BioNLP-ST_2011/train.json distant_supervision/data/BioNLP-ST_2011/train_masked.json
python -m conversion.ds_tag_entities distant_supervision/data/BioNLP-ST_2011/dev.json distant_supervision/data/BioNLP-ST_2011/dev_masked.json
python -m conversion.ds_tag_entities distant_supervision/data/BioNLP-ST_2011/test.json distant_supervision/data/BioNLP-ST_2011/test_masked.json
python -m conversion.ds_tag_entities distant_supervision/data/BioNLP-ST_2011/all.json distant_supervision/data/BioNLP-ST_2011/all_masked.json

python -m conversion.ds_tag_entities distant_supervision/data/BioNLP-ST_2013/train.json distant_supervision/data/BioNLP-ST_2013/train_masked.json
python -m conversion.ds_tag_entities distant_supervision/data/BioNLP-ST_2013/dev.json distant_supervision/data/BioNLP-ST_2013/dev_masked.json
python -m conversion.ds_tag_entities distant_supervision/data/BioNLP-ST_2013/test.json distant_supervision/data/BioNLP-ST_2013/test_masked.json
python -m conversion.ds_tag_entities distant_supervision/data/BioNLP-ST_2013/all.json distant_supervision/data/BioNLP-ST_2013/all_masked.json


## Generate comb-dist data
###

mkdir -p distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2011/
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2011/train.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2011/train.json --direct_data distant_supervision/data/BioNLP-ST_2013/all.json --pair_blacklist distant_supervision/data/BioNLP-ST_2011/dev.json distant_supervision/data/BioNLP-ST_2011/test.json
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2011/dev.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2011/dev.json
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2011/test.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2011/test.json

python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2011/train_masked.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2011/train_masked.json --direct_data distant_supervision/data/BioNLP-ST_2013/all_masked.json --pair_blacklist distant_supervision/data/BioNLP-ST_2011/dev.json distant_supervision/data/BioNLP-ST_2011/test.json
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2011/dev_masked.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2011/dev_masked.json
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2011/test_masked.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2011/test_masked.json

mkdir -p distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2013/
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2013/train.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2013/train.json --direct_data distant_supervision/data/BioNLP-ST_2011/all.json --pair_blacklist distant_supervision/data/BioNLP-ST_2013/dev.json distant_supervision/data/BioNLP-ST_2013/test.json
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2013/dev.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2013/dev.json
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2013/test.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2013/test.json

python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2013/train_masked.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2013/train_masked.json --direct_data distant_supervision/data/BioNLP-ST_2011/all_masked.json --pair_blacklist distant_supervision/data/BioNLP-ST_2013/dev.json distant_supervision/data/BioNLP-ST_2013/test.json
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2013/dev_asked.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2013/dev_masked.json
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/BioNLP-ST_2013/test_masked.json distant_supervision/comb_dist_direct_relex/data/BioNLP-ST_2013/test_masked.json

## Generate PEDL data
###

python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/train.json distant_supervision/data/BioNLP-ST_2011/train.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/dev.json distant_supervision/data/BioNLP-ST_2011/dev.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/test.json distant_supervision/data/BioNLP-ST_2011/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/all.json distant_supervision/data/BioNLP-ST_2011/all.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/train.json distant_supervision/data/BioNLP-ST_2013/train.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/dev.json distant_supervision/data/BioNLP-ST_2013/dev.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/test.json distant_supervision/data/BioNLP-ST_2013/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/all.json distant_supervision/data/BioNLP-ST_2013/all.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/train_masked.json distant_supervision/data/BioNLP-ST_2011/train_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/dev_masked.json distant_supervision/data/BioNLP-ST_2011/dev_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/test_masked.json distant_supervision/data/BioNLP-ST_2011/test_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/all_masked.json distant_supervision/data/BioNLP-ST_2011/all_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/train_masked.json distant_supervision/data/BioNLP-ST_2013/train_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/dev_masked.json distant_supervision/data/BioNLP-ST_2013/dev_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/test_masked.json distant_supervision/data/BioNLP-ST_2013/test_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/all_masked.json distant_supervision/data/BioNLP-ST_2013/all_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

## Generate all data (for the PID run)
###
python -m conversion.combine_ds_data distant_supervision/data/BioNLP-ST_2011/all.json distant_supervision/data/BioNLP-ST_2013/all.json distant_supervision/data/BioNLP-STs/all.json
python -m conversion.combine_ds_data distant_supervision/data/BioNLP-ST_2011/all_masked.json distant_supervision/data/BioNLP-ST_2013/all_masked.json distant_supervision/data/BioNLP-STs/all_masked.json
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-STs/all.json distant_supervision/data/BioNLP-STs/all.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-STs/all_masked.json distant_supervision/data/BioNLP-STs/all_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
