import sys
import configparser
import logging
import copy

import numpy as np
import cv2
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from dataloaders.video_dataset import DecordVideoDataset as VideoDataset
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
    def get_random_crop(image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + crop_height, x: x + crop_width]
        return crop

    image_size = config.getint('ImageSize')

    # Dataset loader
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: cv2.resize(x, dsize=(int(image_size * 1.075), int(image_size * 1.075)), interpolation=cv2.INTER_CUBIC)),
        transforms.Lambda(lambda x: get_random_crop(x, image_size, image_size)),
        transforms.Lambda(lambda x: cv2.flip(x, flipCode=1) if np.random.randint(2) else x),
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

    logging_freq = config.getint('LoggingFreq')

    starting_epoch = config.getint('StartingEpoch')
    epoch_number = config.getint('EpochNumber')
    batches_in_epoch = config.getint('BatchesInEpoch')

    if starting_epoch == 0:
        model.init_networks_normal()
    else:
        model.load_networks(starting_epoch - 1)

    tb_writer = SummaryWriter(f'./runs/experiment-{config.get("ExperimentName")}/')

    logger = Logger(tb_writer, starting_epoch, epoch_number, batches_in_epoch)

    for epoch in range(starting_epoch, epoch_number):
        logger.new_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            losses = model.optimize_parameters(batch)
            if (batch_idx + 1) % logging_freq == 0:
                # log
                logger.end_batch(batch_idx, losses)
                # images
                data = copy.deepcopy(batch)
                tb_writer.add_image('TrainImages/A/real', data['A'][0] * 0.5 + 0.5, epoch)
                tb_writer.add_image('TrainImages/B/real', data['B'][0] * 0.5 + 0.5, epoch)
                data = model.forward(data)
                tb_writer.add_image('TrainImages/A/fake', data['A'][0] * 0.5 + 0.5, epoch)
                tb_writer.add_image('TrainImages/B/fake', data['B'][0] * 0.5 + 0.5, epoch)
        model.save_networks(epoch)

    tb_writer.flush()
    tb_writer.close()


if __name__ == '__main__':
    main()
