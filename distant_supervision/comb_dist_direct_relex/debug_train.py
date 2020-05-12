import json
import shutil
import sys
import os

from allennlp.commands import main

config_file = "allennlp_config/config.json"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": 0},
                        })

serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

os.environ['SEED']='13270'
os.environ['PYTORCH_SEED']='5005'
os.environ['NUMPY_SEED']='5005'


# change the following two variables to make the problem smaller for debugging
os.environ['negative_exampels_percentage']='100' # set to 100 to use all of the dataset. Values < 100 will randomely drop some of the negative examples
os.environ['max_bag_size']='100'  # set to 25 to use all of the dataset. Keep only the top `max_bag_size` sentences in each bag and drop the rest


# reader configurations
os.environ['with_direct_supervision']='false'  # false for distant supervision only


# model configurations
os.environ['dropout_weight'] = '0.1'  # dropout weight after word embeddings
os.environ['with_entity_embeddings']='true'  # false for no entity embeddings

os.environ['sent_loss_weight'] = '1'  # 0, 0.5, 1, 2, 4, 8, 16, 32, 64
os.environ['attention_weight_fn'] = 'sigmoid'  # uniform, softmax, sigmoid, norm_sigmoid
os.environ['attention_aggregation_fn'] = 'max'  # avg, max
os.environ['cnn_size'] = '100'  # avg, max


# trainer configurations
os.environ['batch_size'] = '64'
os.environ['cuda_device'] = '0'  # which GPU to use. Use -1 for no-gpu
os.environ['num_epochs'] = '100'  # set to 100 and rely on early stopping


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "relex",
    "-o", overrides,
]

main()