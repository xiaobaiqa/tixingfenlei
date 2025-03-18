import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO


# 配置参数
class Config :
    data_path = r"C:\datasets\coco2017"
    ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 4  # 减少批大小以适应显存
    num_classes = 5
    img_size = (256, 192)  # (height, width)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")


# 可序列化的转换函数
def resize_image (img) :
    return cv2.resize (img, (Config.img_size [1], Config.img_size [0]))


def horizontal_flip (img) :
    return cv2.flip (img, 1)


def color_jitter (img) :
    img = img.astype (np.float32)
    img += np.random.normal (0, 25, img.shape)
    return np.clip (img, 0, 255).astype (np.uint8)


def normalize_image (img) :
    img = img.astype (np.float32) / 255.0
    img = (img - Config.mean) / Config.std
    return img.transpose (2, 0, 1)  # HWC to CHW


# 数据集类
class CocoBodyDataset (Dataset) :
    def __init__ (self, metadata, train=True) :
        self.metadata = metadata
        self.train = train

    def __len__ (self) :
        return len (self.metadata)

    def __getitem__ (self, idx) :
        item = self.metadata [idx]

        # 读取图像
        img = cv2.imread (item ["img_path"])
        if img is None :
            print (f"无法读取图像：{item ['img_path']}")
            return torch.zeros (3, *Config.img_size), torch.tensor (-1)

        img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
        img = self.apply_transforms (img)

        return torch.FloatTensor (img), torch.tensor (item ["label"], dtype=torch.long)

    def apply_transforms (self, img) :
        # 基础处理
        img = resize_image (img)

        # 数据增强
        if self.train :
            if np.random.rand () > 0.5 :
                img = horizontal_flip (img)
            if np.random.rand () > 0.5 :
                img = color_jitter (img)

        # 标准化
        return normalize_image (img)


# 简单CNN模型
class BodyTypeCNN (nn.Module) :
    def __init__ (self) :
        super ().__init__ ()
        self.features = nn.Sequential (
            nn.Conv2d (3, 32, 3, padding=1),
            nn.ReLU (),
            nn.MaxPool2d (2),
            nn.Conv2d (32, 64, 3, padding=1),
            nn.ReLU (),
            nn.MaxPool2d (2),
            nn.AdaptiveAvgPool2d (1)
        )
        self.classifier = nn.Linear (64, Config.num_classes)

    def forward (self, x) :
        x = self.features (x)
        return self.classifier (x.view (x.size (0), -1))


# 生成元数据
def generate_metadata () :
    if os.path.exists ("metadata.json") :
        with open ("metadata.json") as f :
            return json.load (f)

    coco = COCO (Config.ann_file)
    metadata = []

    for img_id in coco.getImgIds (catIds=coco.getCatIds ('person')) :
        img_info = coco.loadImgs (img_id) [0]
        img_path = os.path.join (Config.img_dir, img_info ["file_name"])

        if not os.path.exists (img_path) :
            continue

        anns = coco.loadAnns (coco.getAnnIds (imgIds=img_id, iscrowd=False))
        for ann in anns :
            if ann ["num_keypoints"] >= 10 :
                kpts = np.array (ann ["keypoints"]).reshape (-1, 3)
                if all (kpts [i, 2] > 0 for i in [5, 6, 11, 12]) :
                    metadata.append ({
                        "img_path" : img_path,
                        "label" : classify_body_type (kpts)
                    })
                    break  # 每图取一个实例

    with open ("metadata.json", "w") as f :
        json.dump (metadata [:1000], f)  # 示例取1000个样本
    return metadata [:1000]


# 体型分类规则
def classify_body_type (kpts) :
    ls = kpts [5] [:2]
    rs = kpts [6] [:2]
    lh = kpts [11] [:2]
    rh = kpts [12] [:2]

    shoulder_width = np.linalg.norm (ls - rs)
    hip_width = np.linalg.norm (lh - rh)
    eps = 1e-6
    shr = shoulder_width / (hip_width + eps)
    whr = (0.7 * shoulder_width) / (hip_width + eps)

    if shr > 1.15 and whr < 0.85 :
        return 0
    elif shr > 1.1 and whr >= 0.9 :
        return 1
    elif shr < 0.95 and whr < 0.98 :
        return 2
    elif 0.98 <= shr <= 1.05 and whr > 0.95 :
        return 3
    else :
        return 4


# 训练流程
def train () :
    metadata = generate_metadata ()
    train_meta, val_meta = train_test_split (metadata, test_size=0.2, random_state=42)

    # 创建数据集
    train_set = CocoBodyDataset (train_meta, train=True)
    val_set = CocoBodyDataset (val_meta, train=False)

    # 数据加载器（Windows需设置num_workers=0）
    train_loader = DataLoader (train_set, batch_size=Config.batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader (val_set, batch_size=Config.batch_size,
                             num_workers=0)

    # 初始化模型
    model = BodyTypeCNN ().to (Config.device)
    criterion = nn.CrossEntropyLoss ()
    optimizer = torch.optim.Adam (model.parameters (), lr=1e-3)

    # 训练循环
    for epoch in range (10) :
        model.train ()
        for inputs, labels in train_loader :
            inputs = inputs.to (Config.device)
            labels = labels.to (Config.device)

            optimizer.zero_grad ()
            outputs = model (inputs)
            loss = criterion (outputs, labels)
            loss.backward ()
            optimizer.step ()

        # 验证
        model.eval ()
        correct = 0
        total = 0
        with torch.no_grad () :
            for inputs, labels in val_loader :
                inputs = inputs.to (Config.device)
                labels = labels.to (Config.device)

                outputs = model (inputs)
                _, predicted = torch.max (outputs.data, 1)
                total += labels.size (0)
                correct += (predicted == labels).sum ().item ()

        print (f"Epoch {epoch + 1}: Val Acc {100 * correct / total:.1f}%")


if __name__ == "__main__" :
    train ()