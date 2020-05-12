#for worker in {0..79}; do
#    OMP_NUM_THREADS=1 nice -n19 python conversion/generate_statistics.py --data "$1" --offsets bioconcepts2pubtatorcentral.offset --out "$1"_raw/statistics.tsv --mapping data/geneid2uniprot.json --species rat,mouse,rabbit,hamster --worker "$worker" --n_workers 80 &
#done

python conversion/aggregate_statistics.py data/PathwayCommons11.pid.hgnc.txt.train.json_raw/ data/PathwayCommons11.pid.hgnc.txt.train.json_raw_statistics.tsv
