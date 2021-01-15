import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
import pickle
from omegaconf import DictConfig
import os
manifest_path = 'data/manifest/'
train_manifest_file = 'train_common_tw_manifest.pkl'
test_manifest_file = 'test_common_tw_manifest.pkl'
TRAIN_MANIFEST_PATH = 'train_manifest_path'
TEST_MANIFEST_PATH = 'test_manifest_path'
MODEL_PATH = 'model/'

config_path = 'examples/asr/conf/quartznet_15x5_zh.yaml'
def load_config(config_path):
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    return params

def set_dataset_path(params, dataset_dict):
    params['model']['train_ds']['manifest_filepath'] = dataset_dict[TRAIN_MANIFEST_PATH]
    params['model']['validation_ds']['manifest_filepath'] = dataset_dict[TEST_MANIFEST_PATH]
    return params

if __name__ == '__main__':
    params = load_config(config_path)
    dataset_dict = {TRAIN_MANIFEST_PATH: os.path.join(manifest_path, train_manifest_file),
                    TEST_MANIFEST_PATH: os.path.join(manifest_path, test_manifest_file)}

    #set training data path
    params = set_dataset_path(params,dataset_dict)

    #load model
    original_quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-Zh")
    original_quartznet.setup_training_data(train_data_config=params['model']['train_ds'])
    original_quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(original_quartznet)

    # save model
    original_quartznet.sve_to(os.path.join(MODEL_PATH,'common_tw_asr.mod'))
