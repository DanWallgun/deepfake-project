import time
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, tb_writer: SummaryWriter, starting_epoch: int, epoch_number: int, batches_in_epoch: int):
        self.writer = tb_writer
        self.starting_epoch = starting_epoch
        self.epoch_number = epoch_number
        self.batches_in_epoch = batches_in_epoch
        self.start_time = time.perf_counter()

    def new_epoch(self, epoch: int):
        self.losses = {}
        self.epoch = epoch
        self.start_epoch_time = time.perf_counter()

    def end_batch(self, batch: int, losses: dict):
        self.batch = batch
        batches_done = self.batches_in_epoch * (self.epoch - self.starting_epoch) + (self.batch + 1)
        batches_left_in_epoch = self.batches_in_epoch - (self.batch + 1)
        batches_left = self.batches_in_epoch * (self.epoch_number - (self.epoch + 1)) + batches_left_in_epoch

        # losses
        for k, v in losses.items():
            if k not in self.losses:
                self.losses[k] = (v, 1)
            else:
                prev = self.losses[k]
                self.losses[k] = (prev[0] + v, prev[1] + 1)
        for loss_name, (loss_sum, prob_count) in self.losses.items():
            self.writer.add_scalar(f'Epoch[{self.epoch}]/{loss_name}', loss_sum/prob_count, batches_done)

        # timing
        elapsed_epoch = time.perf_counter() - self.start_epoch_time
        elapsed = time.perf_counter() - self.start_time
        eta_epoch = elapsed / batches_done * batches_left_in_epoch
        eta = elapsed / batches_done * batches_left
        self.writer.add_text('eta/train', f'ETA[{timedelta(seconds=eta)}]')
        self.writer.add_text('eta/epoch', f'ETA_EPOCH[{timedelta(seconds=eta_epoch)}]')
