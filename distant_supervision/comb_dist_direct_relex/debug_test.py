import json
import shutil
import sys
import os

from allennlp.commands import main


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "runs/run001/model.tar.gz",
    "data/BioNLP-ST_2011/dev.json",
    "--include-package", "relex",
    "--cuda-device", "0",
    "--batch-size", "4",
    "--use-dataset-reader",
    "--predictor", "relex",
    "--output-file", "debug-preds.json",
    "-o", "{dataset_reader: {'with_metadata': true}}",
    "--silent"

]

main()