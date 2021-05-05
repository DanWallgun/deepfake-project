import numpy as np
import cv2
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """
    Provides random access to video file by cv2.VideoCapture
    """
    def __init__(self, video_paths: list, length: int, transform=None):
        assert video_paths is not None and len(video_paths) == 2

        self.cap_A = cv2.VideoCapture(video_paths[0])
        self.len_A = int(self.cap_A.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cap_B = cv2.VideoCapture(video_paths[1])
        self.len_B = int(self.cap_B.get(cv2.CAP_PROP_FRAME_COUNT))

        self.length = length

        self.transform = transform

    def __getitem__(self, index):
        self.cap_A.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(self.len_A))
        _, frame = self.cap_A.read()
        item_A = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        self.cap_B.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(self.len_B))
        _, frame = self.cap_B.read()
        item_B = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return {
            'A': item_A,
            'B': item_B
        }

    def __len__(self):
        return self.length

    def __del__(self):
        self.cap_A.release()
        self.cap_B.release()
