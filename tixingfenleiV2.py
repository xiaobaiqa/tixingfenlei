# -*- coding: utf-8 -*-
"""
优化版人体体型分类系统 v2.0
主要改进：
1. 修正关键点索引逻辑
2. 增强数据验证和标准化
3. 改进模型架构
4. 添加可视化调试功能
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# 配置参数
class Config :
    data_path = r"C:\datasets\coco2017"  # 修改为实际路径
    ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 16
    num_classes = 4
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    required_kpt_ids = [5, 6, 11, 12]  # COCO原始关键点ID
    input_dim = 6  # 增加特征维度


# 数据加载与预处理
class BodyDataset (torch.utils.data.Dataset) :
    def __init__ (self, coco, img_ids, mode='train') :
        self.coco = coco
        self.img_ids = img_ids
        self.mode = mode
        self.kpt_names = {
            0 : "nose", 1 : "left_eye", 2 : "right_eye", 3 : "left_ear", 4 : "right_ear",
            5 : "left_shoulder", 6 : "right_shoulder", 11 : "left_hip", 12 : "right_hip"
        }

        # 初始化标准化参数
        self.mean = torch.FloatTensor([0.0]*Config.input_dim)
        self.std = torch.FloatTensor([1.0]*Config.input_dim)
        if mode == 'train' :
            self._init_normalization ()

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

            # 筛选有效标注
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

            # 处理关键点
            kpts = np.array (main_ann ['keypoints']).reshape (-1, 3)
            visible = kpts [:, 2] > 0
            visible_ids = np.where (visible) [0]

            # 按COCO原始ID排序
            sorted_indices = np.argsort (visible_ids)
            kpts_visible = kpts [visible] [sorted_indices]
            visible_ids_sorted = visible_ids [sorted_indices]

            # 提取特征
            features = self._extract_geo_features (kpts_visible, visible_ids_sorted)
            if features is None :
                return None

            # 标准化处理
            features = torch.FloatTensor (features)  # 新增转换
            features = (features - self.mean) / self.std

            # 生成标签（示例，需替换为真实标注）
            label = self._generate_label (features)

            return torch.FloatTensor (features), torch.tensor (label)
        except Exception as e :
            print (f"Error processing image {img_id}: {str (e)}")
            return None

    def _init_normalization (self) :
        """使用numpy计算后转换为Tensor"""
        features = []
        for img_id in self.img_ids [:1000] :
            item = self._process_item (img_id)
            if item is not None :
                features.append (item [0].numpy ())  # 保持numpy计算

        if features :
            self.mean = torch.FloatTensor (np.mean (features, axis=0))
            self.std = torch.FloatTensor (np.std (features, axis=0))

    def _process_item (self, img_id) :
        """用于标准化参数计算的内部方法"""
        # ...（类似__getitem__处理逻辑但跳过标准化步骤）

    def _extract_geo_features (self, kpts, visible_ids) :
        """增加数值稳定性的特征提取"""
        try :
            id_to_pos = {vid : pos for pos, vid in enumerate (visible_ids)}

            # 验证必要关键点
            required = Config.required_kpt_ids
            if any (rid not in id_to_pos for rid in required) :
                return None

            # 获取坐标点
            ls = kpts [id_to_pos [5], :2]
            rs = kpts [id_to_pos [6], :2]
            lh = kpts [id_to_pos [11], :2]
            rh = kpts [id_to_pos [12], :2]

            # 计算基础几何量（增加极小值保护）
            eps = 1e-6  # 数值稳定性常数

            shoulder_width = np.linalg.norm (ls - rs)
            hip_width = np.linalg.norm (lh - rh)
            torso_height = ((lh [1] + rh [1]) / 2 - (ls [1] + rs [1]) / 2)

            # 添加保护条件
            if hip_width < eps or abs (torso_height) < eps :
                return None

            # 计算特征比率（使用保护后的分母）
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
        """生成模拟标签（实际应用需替换为真实标注）"""
        # 示例逻辑：基于特征聚类生成伪标签
        shr = features [0]
        if shr > 1.2 :
            return 0  # 倒三角
        elif shr < 0.9 :
            return 1  # 苹果型
        else :
            return 3  # 标准型

    def visualize (self, idx) :
        """可视化关键点验证"""
        item = self.__getitem__ (idx)
        if item is None :
            print ("Invalid sample")
            return

        features, label = item
        kpts = self.coco.loadAnns (self.coco.getAnnIds (imgIds=self.img_ids [idx])) [0] ['keypoints']
        kpts = np.array (kpts).reshape (-1, 3)

        plt.figure (figsize=(12, 6))

        # 原始图像
        plt.subplot (121)
        img = cv2.imread (os.path.join (Config.img_dir,
                                        self.coco.loadImgs (self.img_ids [idx]) [0] ['file_name']))
        plt.imshow (cv2.cvtColor (img, cv2.COLOR_BGR2RGB))
        plt.title (f"Label: {label}")

        # 关键点可视化
        plt.subplot (122)
        visible = kpts [:, 2] > 0
        plt.scatter (kpts [visible, 0], kpts [visible, 1], c='r')
        for i in np.where (visible) [0] :
            plt.text (kpts [i, 0], kpts [i, 1], str (i), color='blue')

        # 绘制躯干连线
        for pair in [(5, 6), (5, 11), (6, 12), (11, 12)] :
            if all (kpts [p, 2] > 0 for p in pair) :
                plt.plot (kpts [pair, 0], kpts [pair, 1], 'g-')

        plt.gca ().invert_yaxis ()
        plt.show ()


# 改进的神经网络模型
class EnhancedClassifier (nn.Module) :
    def __init__ (self) :
        super ().__init__ ()
        self.net = nn.Sequential (
            nn.Linear (Config.input_dim, 128),
            nn.BatchNorm1d (128),
            nn.ReLU (),
            nn.Dropout (0.3),
            nn.Linear (128, 64),
            nn.ReLU (),
            nn.Linear (64, Config.num_classes)
        )

    def forward (self, x) :
        return self.net (x)


# 训练流程优化
def train () :
    # 初始化数据集
    coco = COCO (Config.ann_file)
    img_ids = coco.getImgIds (catIds=coco.getCatIds (catNms=['person']))

    # 划分训练测试集
    train_ids, test_ids = train_test_split (img_ids, test_size=0.2, random_state=42)

    # 创建数据集实例
    train_set = BodyDataset (coco, train_ids, 'train')
    test_set = BodyDataset (coco, test_ids, 'test')

    # 数据加载器（增强鲁棒性）
    train_loader = torch.utils.data.DataLoader (
        train_set, batch_size=Config.batch_size, shuffle=True,
        collate_fn=lambda x : [item for item in x if item is not None]
    )
    test_loader = torch.utils.data.DataLoader (
        test_set, batch_size=Config.batch_size,
        collate_fn=lambda x : [item for item in x if item is not None]
    )

    # 初始化模型
    model = EnhancedClassifier ().to (Config.device)
    criterion = nn.CrossEntropyLoss ()
    optimizer = torch.optim.AdamW (model.parameters (), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (optimizer, 'max', patience=3)

    # 训练循环
    best_acc = 0
    for epoch in range (30) :
        model.train ()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader :
            if not batch :
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
            _, predicted = outputs.max (1)
            correct += predicted.eq (labels).sum ().item ()
            total += labels.size (0)

        # 验证集评估
        model.eval ()
        test_correct = 0
        test_total = 0
        with torch.no_grad () :
            for batch in test_loader :
                if batch :
                    features, labels = zip (*batch)
                    features = torch.stack (features).to (Config.device)
                    labels = torch.stack (labels).to (Config.device)

                    outputs = model (features)
                    _, predicted = outputs.max (1)
                    test_correct += predicted.eq (labels).sum ().item ()
                    test_total += labels.size (0)

        # 计算指标
        train_acc = correct / total if total > 0 else 0
        test_acc = test_correct / test_total if test_total > 0 else 0
        avg_loss = total_loss / len (train_loader)

        # 学习率调整
        scheduler.step (test_acc)

        # 保存最佳模型
        if test_acc > best_acc :
            best_acc = test_acc
            torch.save (model.state_dict (), "best_model.pth")

        print (f"Epoch {epoch + 1}: Loss={avg_loss:.4f} | Train Acc={train_acc:.4f} | Test Acc={test_acc:.4f}")

    # 最终评估
    model.load_state_dict (torch.load ("best_model.pth"))
    model.eval ()
    y_true = []
    y_pred = []
    with torch.no_grad () :
        for batch in test_loader :
            if batch :
                features, labels = zip (*batch)
                features = torch.stack (features).to (Config.device)
                labels = torch.stack (labels).cpu ().numpy ()

                outputs = model (features).argmax (1).cpu ().numpy ()
                y_true.extend (labels)
                y_pred.extend (outputs)

    print ("\nClassification Report:")
    print (classification_report (y_true, y_pred, target_names=['倒三角', '苹果型', '不对称型', '标准型']))


if __name__ == "__main__" :
    # 可视化验证
    coco = COCO (Config.ann_file)
    img_ids = coco.getImgIds (catIds=coco.getCatIds (catNms=['person']))
    dataset = BodyDataset (coco, img_ids)
    dataset.visualize (41)  # 随机查看样本

    # 启动训练
    train ()