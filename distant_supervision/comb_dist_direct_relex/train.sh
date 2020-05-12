#!/usr/bin/env bash

allennlp train allennlp_config/config.json --include-package relex -s runs/run020 -o '{"train_data_path": "data/BioNLP-ST_2011/train_masked.json", "validation_data_path": "data/BioNLP-ST_2011/dev_masked.json", "random_seed":  6006, "pytorch_seed": 6006, "numpy_seed": 6006}'
allennlp train allennlp_config/config.json --include-package relex -s runs/run021 -o '{"train_data_path": "data/BioNLP-ST_2011/train_masked.json", "validation_data_path": "data/BioNLP-ST_2011/dev_masked.json", "random_seed":  7007, "pytorch_seed": 7007, "numpy_seed": 7007}'
allennlp train allennlp_config/config.json --include-package relex -s runs/run022 -o '{"train_data_path": "data/BioNLP-ST_2011/train_masked.json", "validation_data_path": "data/BioNLP-ST_2011/dev_masked.json", "random_seed":  8008, "pytorch_seed": 8008, "numpy_seed": 8008}'



allennlp train allennlp_config/config.json --include-package relex -s runs/run023 -o '{"train_data_path": "data/BioNLP-ST_2013/train_masked.json", "validation_data_path": "data/BioNLP-ST_2013/dev_masked.json", "random_seed":  6006, "pytorch_seed": 6006, "numpy_seed": 6006}'
allennlp train allennlp_config/config.json --include-package relex -s runs/run024 -o '{"train_data_path": "data/BioNLP-ST_2013/train_masked.json", "validation_data_path": "data/BioNLP-ST_2013/dev_masked.json", "random_seed":  7007, "pytorch_seed": 7007, "numpy_seed": 7007}'
allennlp train allennlp_config/config.json --include-package relex -s runs/run025 -o '{"train_data_path": "data/BioNLP-ST_2013/train_masked.json", "validation_data_path": "data/BioNLP-ST_2013/dev_masked.json", "random_seed":  8008, "pytorch_seed": 8008, "numpy_seed": 8008}'
