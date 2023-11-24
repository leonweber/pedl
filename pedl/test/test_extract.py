import os.path
import shutil

import pandas as pd
import pytest

from pedl.predict import predict
from hydra import initialize, compose

BASE_PATH = os.path.dirname(os.path.abspath(__file__))



def test_protein_protein(mock_hydra_main):
    out_dir = "resources/outputs/test_protein_protein"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    with initialize(config_path="../configs/"):
        cfg = compose(config_name="predict.yaml")
    with initialize(config_path="../configs/type"):
        cfg_type = compose(config_name="protein_protein.yaml")

    cfg.e1 = "CMTM6"
    cfg.e2 = "CD274"
    cfg.type = cfg_type
    cfg.out = out_dir
    predict(cfg)
    assert os.path.exists(out_dir)
    assert os.path.exists(os.path.join(out_dir, "54918-_-29126.txt"))
    df = pd.read_csv(os.path.join(out_dir, "54918-_-29126.txt"), sep="\t", header=None)
    assert len(df) > 1000
    assert df.iloc[:, 1].sum() > 100


def test_drug_protein(mock_hydra_main):
    out_dir = "resources/outputs/test_drug_protein"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    with initialize(config_path="../configs/"):
        cfg = compose(config_name="predict.yaml")
    with initialize(config_path="../configs/type"):
        cfg_type = compose(config_name="drug_protein.yaml")

    cfg.e1 = "MeSH:D063325"
    cfg.e2 = "1813"
    cfg.use_ids = True
    cfg.type = cfg_type
    cfg.out = out_dir
    predict(cfg)
    assert os.path.exists(out_dir)
    assert os.path.exists(os.path.join(out_dir, "MESH_D063325-_-1813.txt"))
    df = pd.read_csv(os.path.join(out_dir, "MESH_D063325-_-1813.txt"), sep="\t", header=None)
    assert len(df) > 2
    assert df.iloc[:, 1].sum() > 2