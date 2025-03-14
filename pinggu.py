# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


class Config :
    data_path = r"C:\datasets\coco2017"
    ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 16
    num_classes = 4
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    required_kpt_ids = [5, 6, 11, 12]
    input_dim = 6


class BodyDataset (torch.utils.data.Dataset) :
    def __init__ (self, coco, img_ids, mode='train') :
        self.coco = coco
        self.img_ids = img_ids
        self.mode = mode
        self.kpt_names = {
            0 : "nose", 1 : "left_eye", 2 : "right_eye", 3 : "left_ear", 4 : "right_ear",
            5 : "left_shoulder", 6 : "right_shoulder", 11 : "left_hip", 12 : "right_hip"
        }
        self.mean = torch.FloatTensor ([0.0] * Config.input_dim)
        self.std = torch.FloatTensor ([1.0] * Config.input_dim)
        if mode == 'train' :
            self._init_normalization ()

    def __len__ (self) :
        return len (self.img_ids)

    def __getitem__ (self, idx) :
        try :
            img_id = self.img_ids [idx]
            img_info = self.coco.loadImgs (img_id) [0]
            img_path = os.path.join (Config.img_dir, img_info ['file_name'])

            ann_ids = self.coco.getAnnIds (imgIds=img_id)
            anns = self.coco.loadAnns (ann_ids)

            valid_anns = []
            for ann in anns :
                if ann ['iscrowd'] == 0 and ann ['num_keypoints'] >= 10 :
                    kpts = np.array (ann ['keypoints']).reshape (-1, 3)
                    visible = kpts [:, 2] > 0
                    visible_ids = np.where (visible) [0]
                    if all (rid in visible_ids for rid in Config.required_kpt_ids) :
                        valid_anns.append (ann)

            if not valid_anns :
                return None
            main_ann = max (valid_anns, key=lambda x : x ['area'])

            kpts = np.array (main_ann ['keypoints']).reshape (-1, 3)
            visible = kpts [:, 2] > 0
            visible_ids = np.where (visible) [0]

            sorted_indices = np.argsort (visible_ids)
            kpts_visible = kpts [visible] [sorted_indices]
            visible_ids_sorted = visible_ids [sorted_indices]

            features = self._extract_geo_features (kpts_visible, visible_ids_sorted)
            if features is None :
                return None

            features = torch.FloatTensor (features)
            features = (features - self.mean) / self.std

            label = self._generate_label (features)
            return features, torch.tensor (label)
        except Exception as e :
            print (f"Error processing image {img_id}: {str (e)}")
            return None

    def _init_normalization (self) :
        features = []
        for img_id in self.img_ids [:1000] :
            item = self._process_item (img_id)
            if item is not None :
                features.append (item [0].numpy ())
        if features :
            self.mean = torch.FloatTensor (np.mean (features, axis=0))
            self.std = torch.FloatTensor (np.std (features, axis=0))

    def _process_item (self, img_id) :
        try :
            img_info = self.coco.loadImgs (img_id) [0]
            ann_ids = self.coco.getAnnIds (imgIds=img_id)
            anns = self.coco.loadAnns (ann_ids)

            valid_anns = []
            for ann in anns :
                if ann ['iscrowd'] == 0 and ann ['num_keypoints'] >= 10 :
                    kpts = np.array (ann ['keypoints']).reshape (-1, 3)
                    visible = kpts [:, 2] > 0
                    visible_ids = np.where (visible) [0]
                    if all (rid in visible_ids for rid in Config.required_kpt_ids) :
                        valid_anns.append (ann)

            if not valid_anns :
                return None
            main_ann = max (valid_anns, key=lambda x : x ['area'])

            kpts = np.array (main_ann ['keypoints']).reshape (-1, 3)
            visible = kpts [:, 2] > 0
            visible_ids = np.where (visible) [0]

            sorted_indices = np.argsort (visible_ids)
            kpts_visible = kpts [visible] [sorted_indices]
            visible_ids_sorted = visible_ids [sorted_indices]

            features = self._extract_geo_features (kpts_visible, visible_ids_sorted)
            return (torch.FloatTensor (features), 0) if features is not None else None
        except Exception as e :
            print (f"Processing error: {str (e)}")
            return None

    def _extract_geo_features (self, kpts, visible_ids) :
        try :
            id_to_pos = {vid : pos for pos, vid in enumerate (visible_ids)}
            required = Config.required_kpt_ids
            if any (rid not in id_to_pos for rid in required) :
                return None

            ls = kpts [id_to_pos [5], :2]
            rs = kpts [id_to_pos [6], :2]
            lh = kpts [id_to_pos [11], :2]
            rh = kpts [id_to_pos [12], :2]

            eps = 1e-6
            shoulder_width = np.linalg.norm (ls - rs)
            hip_width = np.linalg.norm (lh - rh)
            torso_height = ((lh [1] + rh [1]) / 2 - (ls [1] + rs [1]) / 2)

            if hip_width < eps or abs (torso_height) < eps :
                return None

            shoulder_hip_ratio = shoulder_width / (hip_width + eps)
            upper_lower_ratio = shoulder_width / (abs (torso_height) + eps)
            left_symmetry = np.linalg.norm (ls - rh)
            right_symmetry = np.linalg.norm (rs - lh)

            return np.array ([
                shoulder_hip_ratio,
                upper_lower_ratio,
                left_symmetry / (hip_width + eps),
                right_symmetry / (hip_width + eps),
                (ls [0] - lh [0]) / (hip_width + eps),
                (rs [0] - rh [0]) / (hip_width + eps)
            ], dtype=np.float32)
        except Exception as e :
            print (f"Feature error: {str (e)}")
            return None

    def _generate_label (self, features) :
        """改进后的标签生成逻辑"""
        shr = features [0].item ()  # 转换为Python标量值
        symmetry_diff = abs (features [2].item () - features [3].item ())

        if shr > 1.2 :
            return 0  # 倒三角型
        elif shr < 0.9 :
            return 1  # 苹果型
        elif symmetry_diff > 0.15 :  # 增加对称性判断
            return 2  # 不对称型
        else :
            return 3  # 标准型


class EnhancedClassifier (torch.nn.Module) :
    def __init__ (self) :
        super ().__init__ ()
        self.net = torch.nn.Sequential (
            torch.nn.Linear (Config.input_dim, 128),
            torch.nn.BatchNorm1d (128),
            torch.nn.ReLU (),
            torch.nn.Dropout (0.3),
            torch.nn.Linear (128, 64),
            torch.nn.ReLU (),
            torch.nn.Linear (64, Config.num_classes)
        )

    def forward (self, x) :
        return self.net (x)


def evaluate (model_path) :
    # 初始化数据集
    coco = COCO (Config.ann_file)
    all_img_ids = coco.getImgIds (catIds=coco.getCatIds (catNms=['person']))
    train_ids, test_ids = train_test_split (all_img_ids, test_size=0.2, random_state=42)

    # 继承训练集的标准化参数
    train_set = BodyDataset (coco, train_ids, 'train')
    test_set = BodyDataset (coco, test_ids, 'test')
    test_set.mean = train_set.mean
    test_set.std = train_set.std

    # 创建测试集加载器
    test_loader = DataLoader (
        test_set,
        batch_size=Config.batch_size,
        collate_fn=lambda x : [item for item in x if item is not None]
    )

    # 加载模型
    model = EnhancedClassifier ().to (Config.device)
    model.load_state_dict (torch.load (model_path))
    model.eval ()

    # 执行评估
    all_preds = []
    all_labels = []
    with torch.no_grad () :
        for batch in test_loader :
            if not batch :
                continue
            features, labels = zip (*batch)
            features = torch.stack (features).to (Config.device)
            outputs = model (features)
            _, preds = torch.max (outputs, 1)
            all_preds.extend (preds.cpu ().numpy ())
            all_labels.extend (torch.stack (labels).cpu ().numpy ())

    # 生成报告（修复后的分类报告）
    print ("================ 模型评估报告 ================")
    print (f"测试样本数量: {len (all_labels)}")
    print (f"类别分布: {np.bincount (all_labels, minlength=4)}")  # 强制显示4个类别

    print ("\n分类性能指标:")
    print (classification_report (
        all_labels, all_preds,
        labels=[0, 1, 2, 3],  # 显式指定所有类别
        target_names=['倒三角', '苹果型', '不对称型', '标准型'],
        digits=4
    ))


if __name__ == "__main__" :
    evaluate ("best_model.pth")