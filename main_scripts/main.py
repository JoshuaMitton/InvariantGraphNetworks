import os
from data_loader.data_generator import DataGenerator
from models.invariant_basic import invariant_basic
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs

from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
#     import tensorflow as tf
    import torch

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
#     gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#     gpuconfig.gpu_options.visible_device_list = config.gpus_list
#     gpuconfig.gpu_options.allow_growth = True
#     sess = tf.Session(config=gpuconfig)
    if config.cuda:
        print(f'Using GPU : {torch.cuda.get_device_name(int(config.gpu))}')
    else:
        print(f'Using CPU')
    # create your data generator
    data = DataGenerator(config)
#     data = torch.from_numpy(data)

    # create an instance of the model you want
    model = invariant_basic(config, data)
    if config.cuda:
        model = model.cuda()
        
    for name, param in model.named_parameters():
#         if param.device.type != 'cuda':
        print(f'{name}, device type {param.device.type}')
        
    # create trainer and pass all the previous components to it
#     trainer = Trainer(sess, model, data, config)
    trainer = Trainer(model, data, config)
    # load model if exists
#     model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
