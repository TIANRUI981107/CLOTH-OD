"""
Please note that this transform is NOT `GeneralizedRCNNTransfrom` in `network_files/transform.py`.
It's used in `datasets preparation`, i.e., horizontal flip images and labels.
"""
import random
from torchvision.transforms import functional as F


class Compose(object):
    """Compose multiple transformation functions"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """Convert PIL to Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """
    Special RandomHorizontalFlip for *Object Detection*.

    For OD task, the bbox must be flipped when we flip its image.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        # if generalized probability from `random module` is less than the given prob.,
        # then flip the image and label.
        if random.random() < self.prob:
            height, width = image.shape[-2:]  # get image's height and width
            image = image.flip(-1)  # horizontal flip the image
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # flip bbox coordinates,
            target["boxes"] = bbox
        return image, target
