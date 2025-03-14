# -*- coding: utf-8 -*-
"""
修正版男性体型分类系统 v1.1
主要修复：
1. COCO关键点索引错误
2. 增强数据验证逻辑
3. 改进异常处理
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# 配置参数
class Config :
    data_path = r"C:\datasets\coco2017"  # 修改为实际路径
    ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 32
    num_classes = 4  # 体型类别数
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    required_kpt_indices = [5, 6, 11, 12]  # 左肩、右肩、左髋、右髋


# 数据加载与预处理
class BodyDataset (torch.utils.data.Dataset) :
    def __init__ (self, coco, img_ids) :
        self.coco = coco
        self.img_ids = img_ids
        self.kpt_names = {
            0 : "nose", 1 : "left_eye", 2 : "right_eye", 3 : "left_ear", 4 : "right_ear",
            5 : "left_shoulder", 6 : "right_shoulder", 11 : "left_hip", 12 : "right_hip"
        }

    def __len__ (self) :
        return len (self.img_ids)

    def __getitem__ (self, idx) :
        try :
            img_id = self.img_ids [idx]
            img_info = self.coco.loadImgs (img_id) [0]
            img_path = os.path.join (Config.img_dir, img_info ['file_name'])

            # 加载图像和标注
            img = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)
            ann_ids = self.coco.getAnnIds (imgIds=img_id)
            anns = self.coco.loadAnns (ann_ids)

            # 选择最大且包含必要关键点的人体实例
            valid_anns = []
            for ann in anns :
                if ann ['iscrowd'] == 0 and ann ['num_keypoints'] >= 10 :
                    kpts = np.array (ann ['keypoints']).reshape (-1, 3)
                    visible = kpts [:, 2] > 0
                    visible_indices = np.where (visible) [0]
                    # 检查是否包含所有必要关键点
                    if all (i in visible_indices for i in Config.required_kpt_indices) :
                        valid_anns.append (ann)

            if not valid_anns :
                return None
            main_ann = max (valid_anns, key=lambda x : x ['area'])

            # 提取关键点并确保顺序
            kpts = np.array (main_ann ['keypoints']).reshape (-1, 3)
            visible = kpts [:, 2] > 0
            visible_indices = np.where (visible) [0]  # 可见关键点的原始ID

            # 过滤不可见点并保留坐标
            kpts_visible = kpts [visible, :2]

            # 按原始关键点ID排序
            sorted_order = np.argsort (visible_indices)
            kpts_sorted = kpts_visible [sorted_order]
            original_ids = visible_indices [sorted_order]  # 排序后的原始ID列表

            # 提取几何特征（传入排序后的关键点和原始ID）
            features = self._extract_geo_features (kpts_sorted, original_ids)
            if features is None :
                return None

            # 生成体型标签
            label = self._assign_body_type (features)

            return torch.FloatTensor (features), torch.tensor (label)
        except Exception as e :
            print (f"Error processing image {img_id}: {str (e)}")
            return None

    def _extract_geo_features (self, kpts, original_ids) :
        """提取几何特征（基于原始关键点ID定位）"""
        try :
            # 定义所需关键点的COCO ID
            required_ids = [5, 6, 11, 12]

            # 检查是否所有必需关键点都存在
            for req_id in required_ids :
                if req_id not in original_ids :
                    print (f"Missing required keypoint ID: {req_id}")
                    return None

            # 获取各个关键点的索引
            idx_map = {
                req_id : np.where (original_ids == req_id) [0] [0]
                for req_id in required_ids
            }

            left_shoulder = kpts [idx_map [5]]
            right_shoulder = kpts [idx_map [6]]
            left_hip = kpts [idx_map [11]]
            right_hip = kpts [idx_map [12]]

            # 计算特征
            shoulder_width = np.linalg.norm (left_shoulder - right_shoulder)
            hip_width = np.linalg.norm (left_hip - right_hip)
            torso_height = (left_hip [1] + right_hip [1]) / 2 - (left_shoulder [1] + right_shoulder [1]) / 2

            # 处理分母为零的情况
            if hip_width == 0 or torso_height == 0 :
                print ("Invalid feature: division by zero")
                return None

            return [
                shoulder_width / hip_width,
                shoulder_width / torso_height,
                (left_shoulder [0] - left_hip [0]) / hip_width,
                (right_shoulder [0] - right_hip [0]) / hip_width
            ]
        except Exception as e :
            print (f"Error in feature extraction: {str (e)}")
            return None

    def _assign_body_type (self, features) :
        """更稳健的体型分类规则"""
        if features is None :
            return 3  # 默认类别

        shr, torso_ratio, left_ratio, right_ratio = features

        # 有效性检查
        if any (np.isnan (x) for x in [shr, torso_ratio]) :
            return 3

        if shr > 1.15 and torso_ratio > 0.8 :
            return 0  # 倒三角
        elif shr < 0.95 and torso_ratio < 0.6 :
            return 1  # 苹果型
        elif abs (left_ratio - right_ratio) > 0.15 :
            return 2  # 不对称型
        else :
            return 3  # 标准型


# 定义分类模型（保持不变）
class BodyClassifier (nn.Module) :
    def __init__ (self, input_dim=4, num_classes=4) :
        super ().__init__ ()
        self.fc = nn.Sequential (
            nn.Linear (input_dim, 64),
            nn.ReLU (),
            nn.Dropout (0.2),
            nn.Linear (64, num_classes)
        )

    def forward (self, x) :
        return self.fc (x)


# 训练流程（改进数据加载）
def train () :
    # 初始化COCO API
    coco = COCO (Config.ann_file)
    img_ids = coco.getImgIds (catIds=coco.getCatIds (catNms=['person']))

    # 过滤有效样本
    print ("Filtering valid samples...")
    valid_ids = []
    for idx, img_id in enumerate (img_ids [:4000]) :  # 限制样本数量
        ann_ids = coco.getAnnIds (imgIds=img_id)
        anns = coco.loadAnns (ann_ids)
        if any (ann ['num_keypoints'] >= 10 for ann in anns) :
            valid_ids.append (img_id)
        if (idx + 1) % 500 == 0 :
            print (f"Processed {idx + 1} images, valid: {len (valid_ids)}")

    # 划分训练测试集
    train_ids, test_ids = train_test_split (valid_ids, test_size=0.2, random_state=42)

    # 创建数据集
    train_set = BodyDataset (coco, train_ids)
    test_set = BodyDataset (coco, test_ids)

    # 创建数据加载器（改进无效样本处理）
    train_loader = torch.utils.data.DataLoader (
        train_set, batch_size=Config.batch_size, shuffle=True,
        collate_fn=lambda x : [item for item in x if item is not None]
    )
    test_loader = torch.utils.data.DataLoader (
        test_set, batch_size=Config.batch_size,
        collate_fn=lambda x : [item for item in x if item is not None]
    )

    # 初始化模型
    model = BodyClassifier ().to (Config.device)
    criterion = nn.CrossEntropyLoss ()
    optimizer = torch.optim.Adam (model.parameters (), lr=0.001)

    # 训练循环
    print ("\nStart training...")
    for epoch in range (20) :
        model.train ()
        total_loss = 0
        valid_batches = 0

        for batch in train_loader :
            if not batch :  # 跳过空批次
                continue

            features, labels = zip (*batch)
            features = torch.stack (features).to (Config.device)
            labels = torch.stack (labels).to (Config.device)

            optimizer.zero_grad ()
            outputs = model (features)
            loss = criterion (outputs, labels)
            loss.backward ()
            optimizer.step ()

            total_loss += loss.item ()
            valid_batches += 1

        # 测试评估
        model.eval ()
        test_features, test_labels = [], []
        with torch.no_grad () :
            for batch in test_loader :
                if batch :
                    f, l = zip (*batch)
                    test_features.append (torch.stack (f))
                    test_labels.append (torch.stack (l))

            if test_features :
                test_features = torch.cat (test_features).to (Config.device)
                test_labels = torch.cat (test_labels).to (Config.device)

                preds = model (test_features).argmax (dim=1)
                acc = (preds == test_labels).float ().mean ().item ()
            else :
                acc = 0.0

        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
        print (f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    # 输出分类报告
    if test_features is not None :
        print ("\nFinal Classification Report:")
        print (classification_report (test_labels.cpu (), preds.cpu ()))
    # 训练结束后保存模型
    torch.save(model.state_dict(), "body_classifier.pth")
    print(f"Model saved to body_classifier.pth")

if __name__ == "__main__" :
    train ()