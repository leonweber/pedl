#!/usr/bin/env bash
runs=( run001 )
#data=PathwayCommons11.pid.hgnc.txt
data=BioNLP-ST_2011
type=test

for run in ${runs[@]}; do
python -m distant_supervision.predict_pedl distant_supervision/data/$data/"$type"_masked.hdf5 distant_supervision/runs/$run/"$data"_"$type"_preds.txt --model_path distant_supervision/runs/$run --data distant_supervision/data/$data/"$type".json --device cuda;
done