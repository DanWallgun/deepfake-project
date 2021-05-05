import time
import configparser
import sys

import cv2
import numpy as np
import tqdm
import torch
import torchvision.transforms as transforms

from models.cycle_gan import CycleGAN
from util.storage import (
    Storage,
    MemoryStorage,
    S3BucketStorage
)
from util.yacloud import s3 as YaS3


def create_config():
    config_parser = configparser.ConfigParser()
    config_parser.read('config.ini')
    config = config_parser['Inference']
    return config


def create_storage(config, name) -> Storage:
    if name + 'Bucket' in config:
        bucket_name = config[name + 'Bucket']
        return S3BucketStorage(YaS3.Bucket(bucket_name))
    elif name + 'Directory' in config:
        directory = config[name + 'Directory']
        return MemoryStorage(directory)
    assert False


def prepare_data(config, datasets_storage):
    path = config.get('InferenceVideo')
    file_bytes = datasets_storage.load_file(path)
    with open(path, 'wb') as f:
        f.write(file_bytes)


def main():
    config = create_config()
    datasets_storage = create_storage(config, 'Datasets')
    prepare_data(config, datasets_storage)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    video_path = config.get('InferenceVideo')
    kek = video_path.split('.')
    kek[-2] += '-out'
    out_video_path = '.'.join(kek)

    torch.autograd.set_grad_enabled(False)

    model = CycleGAN(config, create_storage(config, 'Checkpoints'))
    model.load_networks(config.getint('InferenceEpoch'))
    vid_cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_cap = cv2.VideoWriter(out_video_path, fourcc, 24.0, (512,  256))

    frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 0

    start_time = time.perf_counter()

    buffer = np.ndarray(shape=(0,256,256,3), dtype=np.uint8)
    buffer_tensor = torch.empty((0,3,256,256))

    while vid_cap.isOpened():
        ret, frame = vid_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer = np.append(buffer, [frame], axis=0)
            buffer_tensor = torch.cat((buffer_tensor, transform(frame).view(-1, 3, 256, 256)), dim=0)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        if buffer.shape[0] == 24 * 4 or (not ret and buffer.shape[0] > 0):
            image = buffer_tensor
            image = model.forward({'A': buffer_tensor})['A']
            image = 0.5 * (image + 1.0)
            processed = image.permute(0, 2, 3, 1).cpu().numpy()
            processed = (processed * 255).astype(np.uint8)

            frame_buffer = np.concatenate((buffer, processed), axis=2)
            for frame in frame_buffer:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out_cap.write(frame)

            counter += buffer.shape[0]
            elapsed_time = time.perf_counter() - start_time
            sys.stdout.write('\n' + f'frames = {counter}\ttime = {elapsed_time}\tspeed = {counter / elapsed_time}\tETA = {(frame_count - counter) / (counter / elapsed_time) / 60}')

            buffer = np.ndarray(shape=(0,256,256,3), dtype=np.uint8)
            buffer_tensor = torch.empty((0,3,256,256))

        if not ret:
            break

    vid_cap.release()
    out_cap.release()

    torch.autograd.set_grad_enabled(True)


if __name__ == '__main__':
    main()
