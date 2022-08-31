import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOCDataSet(Dataset):
    """Parsing PASCAL VOC2007/2012 datasets"""

    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt",
                 provided_cls_path=False):
        """
        Args:
            voc_root: root to VOC datasets, e.g. "VOC2007"
            year: specific for VOC datasets, e.g. "2007", "2012"
            transforms: pre-processing method for datasets
            txt_name: initialized datasets, e.g. "train.txt", "val.txt"
            provided_cls_path: whether the `cls_indices` is provided under datasets root dir, e.g.
                at "VOC2007/VOC2007-trainval/cls_indices.json"
        """
        # whether datasets exist, then get `Path to JPEGImages` and `Path to Annotations`
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        if "VOC2007" in voc_root:
            self.root = os.path.join(voc_root, f"VOC{year}-trainval")  # PASCAL VOC datasets
        elif "CLOTH-OD" in voc_root:
            self.root = os.path.join(voc_root, "VOC2007-trainval")  # CLOTH datasets
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")  # raw PASCAL VOC datasets
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt | val.txt, then store all the `Annotations` path
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)
        with open(txt_path) as read:  # open txt_name file (train/train_val/val)
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in read.readlines() if len(line.strip()) > 0]  # get corresponding annotations path

        self.xml_list = []  # store all corresponding annotations path
        objects_list = []   # store all objects(List[Dict[]]) by parsing XML
        obj_uniq_cls = []   # store unique object classes
        for xml_path in xml_list:
            # whether XML file exists
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue

            # whether object targets exist in XML file
            # if existing, store obj_name_per_img
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            else:  # get all objects' cls_indices.json
                obj_cls_per_img = [obj["name"] for obj in data["object"]]

            objects_list.extend(obj_cls_per_img)  # concatenate all the obj_per_img
            obj_uniq_cls = sorted(list(set(objects_list)))  # uniq(objects), sorted in alphabetic order
            self.xml_list.append(xml_path)  # store XML list that are neither empty nor no-objects

        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)

        if not provided_cls_path:
            # get objects_cls_dict
            self.class_dict = dict((cls, idx) for idx, cls in enumerate(obj_uniq_cls, start=1))
            # write `class_dict` to root dir
            with open(os.path.join(self.root, "cls_indices.json"), "w") as cls_json:
                json.dump(self.class_dict, cls_json, indent=4)
        else:
            # read class_indict
            json_file = os.path.join(self.root, "cls_indices.json")
            assert os.path.exists(json_file), "{} file not exist.".format(json_file)
            with open(json_file, 'r') as f:
                self.class_dict = json.load(f)

        # datasets pre-processing method
        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # get all objects' 4 coordinates
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])  # get image path in JPEGImage dir with filename
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []  # whether difficulty to detect, 0 for easy, and 1 for difficulty
        assert "object" in data, "{} lack of object information.".format(xml_path)
        # * ------------------------------- *
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # double-check whether `width` OR `height` <= 0, which will cause `Loss_reg` == Nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' XML, there are some bbox w/h <=0".format(xml_path))
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])  # if multiple boxes exist, `boxes` is a 2-D matrix
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert object's information into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        # width = x_max - x_min, height = y_max - y_min
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        """
        Get image's size.

        `[pytorch tutorial:](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset)`
        "If you want to use aspect ratio grouping during training(so that each batch only contains images with similar
        aspect ratios), then it is recommended to also implementation a `get_height_and_width` method. This method
        returns height and width of the image. If this method is not provided, we query all elements of the dataset
        via `__getitem__`, which loads the image in the memory and is slower than if a custom method is provided."
        """

        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        Parse `xml` into `dict` python object.
        References to `tensorflow` --> `recursive_parse_xml_to_dict`

        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # whether len_of 'XML file contents' == 0
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # recursively loop through secondary tags
            if child.tag != 'object':  # whether tag's name == 'object'
                result[child.tag] = child_result[child.tag]  # write tags and info into Python dict
            else:
                if child.tag not in result:  # Thanks to multiple objects in the same image, we need Python List
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        """
        Collate List[Tuple[Tensor, Dict],] into Tuple[List[Tensor,], List[Dict,]]

        In the image classification task, we don't have to use this method. This is because `my_dataset.py` returns
        `data` with the same img_size, `Pytorch` will simply `torch.stack()` Images into mini-batches for training.

        For this task, `collate_fn` method groups `images` together and `targets` likewise. Later, the `transform`
        will concatenate `Images` and `Labels`  in to List[Tensors,] and List[dicts,] for mini-batch training.

        Args:
            batch: List[Tuples], (sample_1, sample_2, ..., sample_N),
                where N = `batch_size` in `dataloader` and `sample_x` is of tuple type.
                Specifically, `sample_x` is tuple(image[3-D tensor dtype for RGB image], targets[dict dtype])

        Returns:
            Tuple[List[Tensor,], List[Dict,]],
            ((img_1, img_2, ..., img_N], [target_1, target_2, ..., target_N])

        """
        return tuple(zip(*batch))


if __name__ == '__main__':
    import transforms  # customized OD transformation functions `transforms.py`
    from draw_box_utils import draw_objs
    from PIL import Image
    import json
    import matplotlib.pyplot as plt
    import torchvision.transforms as ts  # original Pytorch `torchvision.transforms`
    import random
    from pathlib import Path

    # read class_indices
    category_index = {}
    try:
        json_file = open(Path().joinpath("..", "CLOTH", "VOC2007-trainval", 'cls_indices.json'), 'r')
        class_dict = json.load(json_file)
        category_index = {str(v): str(k) for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)

    # dataset transformation
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # load train data set
    train_data_set = VOCDataSet(os.path.join(os.getcwd(), '..', 'CLOTH'), "2007", data_transform["train"], "train.txt")
    print(f"len_of training datasets: {len(train_data_set)}")
    # loop through 5 images, sampled from `train_data_set`
    for index in random.sample(range(0, len(train_data_set)), k=5):
        img, target = train_data_set[index]  # `__getitem__` dunder method
        img = ts.ToPILImage()(img)
        plot_img = draw_objs(img,
                             target["boxes"].numpy(),
                             target["labels"].numpy(),
                             np.ones(target["labels"].shape[0]),
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        if not os.path.exists("./output"):
            os.makedirs("./output")
        plt.savefig(f"./output/{index}.png", bbox_inches='tight')
        plt.show()
