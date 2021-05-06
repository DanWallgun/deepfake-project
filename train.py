import sys
import configparser
import logging

import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from dataloaders.video_dataset import VideoDataset
from models.cycle_gan import CycleGAN
from util.storage import (
    Storage,
    MemoryStorage,
    S3BucketStorage
)
from util.yacloud import s3 as YaS3
from util.logger import Logger

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def create_config():
    config_parser = configparser.ConfigParser()
    config_parser.read('config.ini')
    config = config_parser['Train']
    return config


def create_storage(config, name) -> Storage:
    if name + 'Bucket' in config:
        bucket_name = config[name + 'Bucket']
        return S3BucketStorage(YaS3.Bucket(bucket_name))
    elif name + 'Directory' in config:
        directory = config[name + 'Directory']
        return MemoryStorage(directory)
    assert False


def create_dataloader(config, datasets_storage, copy_data_to_local=False):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    video_paths = [
        config.get('TrainVideoA'),
        config.get('TrainVideoB')
    ]
    if copy_data_to_local:
        for path in video_paths:
            file_bytes = datasets_storage.load_file(path)
            with open(path, 'wb') as f:
                f.write(file_bytes)
    dataloader = DataLoader(
        VideoDataset(
            video_paths=video_paths,
            length=config.getint('BatchesInEpoch'),
            transform=train_transform
        ),
        batch_size=1,
        shuffle=True
    )
    return dataloader


def main():
    config = create_config()
    datasets_storage = create_storage(config, 'Datasets')
    dataloader = create_dataloader(
        config,
        datasets_storage,
        config.getboolean('CopyDatasetsToLocal')
    )
    model = CycleGAN(config, create_storage(config, 'Checkpoints'))

    starting_epoch = config.getint('StartingEpoch')
    epoch_number = config.getint('EpochNumber')
    batches_in_epoch = config.getint('BatchesInEpoch')

    if starting_epoch == 0:
        model.init_networks_normal()
    else:
        model.load_networks(starting_epoch - 1)

    logger = Logger(
        SummaryWriter(f'./runs/experiment-{config.get("ExperimentName")}/'),
        starting_epoch, epoch_number, batches_in_epoch
    )

    for epoch in range(starting_epoch, epoch_number):
        logger.new_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            losses = model.optimize_parameters(batch)

            logger.end_batch(batch_idx, losses)
        model.save_networks(epoch)

    tb_writer.flush()
    tb_writer.close()


if __name__ == '__main__':
    main()
