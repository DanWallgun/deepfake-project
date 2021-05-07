import numpy as np
import cv2
import torch
from torchvision import transforms

from external.attention.model import BiSeNet


class AttentionProvider():
    def __init__(self, save_path: str, device):
        self.device = device
        self.model = BiSeNet(n_classes=19).to(self.device)
        self.model.load_state_dict(torch.load(
            save_path,
            map_location=self.device
        ))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_weights(self, input_image: torch.Tensor):
        h = input_image.shape[1]
        w = input_image.shape[2]
        image = cv2.resize(
            input_image.permute(1, 2, 0).cpu().numpy(),
            (512, 512),
            interpolation=cv2.INTER_CUBIC
        )
        image = self.transform(image)
        image.requires_grad = False
        image = torch.unsqueeze(image, 0)
        image = image.to(self.device)

        with torch.no_grad():
            out = self.model(image)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        parsing_anno = parsing.copy().astype(np.uint8)
        weights = np.zeros(parsing_anno.shape)

        class_to_weight = {
            # 0: 1,
            # 1: 1,
            # 2: 1,
            # 3: 1,
            4: 1, # also eyes?
            5: 1, # also eyes?
            # 6: 1, # eyes
            # 7: 1, # right ear
            # 8: 1, # left ear
            # 9: 1, # ???
            # 10: 1, # nose
            11: 1, # mouth
            12: 1, # mouth
            13: 1 # mouth
        }

        for cl, wei in class_to_weight.items():
            index = np.where(parsing_anno == cl)
            weights[index[0], index[1]] = wei

        return torch.tensor(cv2.resize(
            weights,
            (h, w),
            interpolation=cv2.INTER_NEAREST
        )).to(self.device)
