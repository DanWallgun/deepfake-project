import sys
import time
import configparser
from datetime import timedelta
import logging

import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataloaders.video_dataset import VideoDataset
from models.cycle_gan import CycleGAN
from util.storage import (
    Storage,
    MemoryStorage,
    S3BucketStorage
)
from util.yacloud import s3 as YaS3

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


def create_dataloader(config, datasets_storage):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    video_paths = [
        config.get('TrainVideoA'),
        config.get('TrainVideoB')
    ]
    #already_downloaded
    #for path in video_paths:
    #     file_bytes = datasets_storage.load_file(path)
    #     with open(path, 'wb') as f:
    #         f.write(file_bytes)
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
    dataloader = create_dataloader(config, datasets_storage)
    model = CycleGAN(config, create_storage(config, 'Checkpoints'))

    starting_epoch = config.getint('StartingEpoch')
    epoch_number = config.getint('EpochNumber')
    batches_in_epoch = config.getint('BatchesInEpoch')

    if starting_epoch == 0:
        model.init_networks_normal()
    else:
        model.load_networks(starting_epoch - 1)

    start_time = time.perf_counter()
    for epoch in range(starting_epoch, epoch_number):
        logger = setup_logger(f'Train:Epoch{epoch}', f'logs/info_log_epoch{epoch}.log')
        for batch_idx, batch in enumerate(dataloader):
            losses = model.optimize_parameters(batch)

            # log
            full_log_str = 'Epoch %03d/%03d [%04d/%04d] -- ' % (epoch + 1, epoch_number, batch_idx + 1, batches_in_epoch)
            for i, loss_name in enumerate(losses.keys()):
                if i + 1 == len(losses.keys()):
                    full_log_str += '%s: %.4f -- ' % (loss_name, losses[loss_name])
                else:
                    full_log_str += '%s: %.4f | ' % (loss_name, losses[loss_name])

            batches_done = batches_in_epoch * (epoch - starting_epoch) + (batch_idx + 1)
            batches_left = batches_in_epoch * (epoch_number - epoch - 1) + batches_in_epoch - (batch_idx + 1)
            elapsed = time.perf_counter() - start_time
            eta_seconds = elapsed / batches_done * batches_left
            full_log_str += 'ETA: %s' % (timedelta(seconds=eta_seconds))
            # print(full_log_str)
            logger.info(full_log_str)
            #####
        model.save_networks(epoch)


if __name__ == '__main__':
    main()
