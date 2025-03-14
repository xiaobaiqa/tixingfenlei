# -*- coding: utf-8 -*-
"""
男性体型分类系统 v1.2
改进内容：
1. 修复Config类定义问题
2. 增强特征验证可视化
3. 添加异常处理
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
import seaborn as sns
from sklearn.manifold import TSNE


# 配置参数（必须首先定义）
class Config :
    data_path = r"C:\datasets\coco2017"  # 使用原始字符串处理Windows路径
    ann_file = os.path.join (data_path, "annotations", "person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 32
    num_classes = 4
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    required_kpt_indices = [5, 6, 11, 12]  # 左肩、右肩、左髋、右髋

    def __init__ (self) :
        self._validate_paths ()

    def _validate_paths (self) :
        """路径验证"""
        if not os.path.exists (self.data_path) :
            raise FileNotFoundError (f"数据集根目录不存在: {self.data_path}")
        if not os.path.exists (self.ann_file) :
            raise FileNotFoundError (f"标注文件不存在: {self.ann_file}")
        if not os.path.exists (self.img_dir) :
            raise FileNotFoundError (f"图像目录不存在: {self.img_dir}")


# 数据加载与预处理
class BodyDataset (torch.utils.data.Dataset) :
    def __init__ (self, coco, img_ids) :
        self.coco = coco
        self.img_ids = img_ids
        self.kpt_names = {
            0 : "nose", 1 : "left_eye", 2 : "right_eye",
            3 : "left_ear", 4 : "right_ear",
            5 : "left_shoulder", 6 : "right_shoulder",
            11 : "left_hip", 12 : "right_hip"
        }

    def __len__ (self) :
        return len (self.img_ids)

    def __getitem__ (self, idx) :
        try :
            img_id = self.img_ids [idx]
            img_info = self.coco.loadImgs (img_id) [0]
            img_path = os.path.join (Config.img_dir, img_info ['file_name'])

            # 加载图像
            if not os.path.exists (img_path) :
                print (f"图像文件缺失: {img_path}")
                return None
            img = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)

            # 处理标注
            ann_ids = self.coco.getAnnIds (imgIds=img_id)
            anns = self.coco.loadAnns (ann_ids)

            # 筛选有效标注
            valid_anns = []
            for ann in anns :
                if ann ['iscrowd'] == 0 and ann ['num_keypoints'] >= 10 :
                    kpts = np.array (ann ['keypoints']).reshape (-1, 3)
                    visible = kpts [:, 2] > 0
                    visible_indices = np.where (visible) [0]
                    if all (i in visible_indices for i in Config.required_kpt_indices) :
                        valid_anns.append (ann)

            if not valid_anns :
                return None

            # 选择最大实例
            main_ann = max (valid_anns, key=lambda x : x ['area'])

            # 处理关键点
            kpts = np.array (main_ann ['keypoints']).reshape (-1, 3)
            visible = kpts [:, 2] > 0
            visible_indices = np.where (visible) [0]

            # 排序关键点
            sorted_order = np.argsort (visible_indices)
            kpts_visible = kpts [visible, :2]
            kpts_sorted = kpts_visible [sorted_order]
            original_ids = visible_indices [sorted_order]

            # 提取特征
            features = self._extract_geo_features (kpts_sorted, original_ids)
            if features is None :
                return None

            # 生成标签
            label = self._assign_body_type (features)

            return torch.FloatTensor (features), torch.tensor (label)

        except Exception as e :
            print (f"处理图像 {img_id} 时出错: {str (e)}")
            return None

    def _extract_geo_features (self, kpts, original_ids) :
        """几何特征提取（带安全校验）"""
        try :
            required_ids = Config.required_kpt_indices
            epsilon = 1e-6

            # 检查关键点完整性
            missing = [rid for rid in required_ids if rid not in original_ids]
            if missing :
                print (f"缺失关键点: {missing}")
                return None

            # 建立索引映射
            idx_map = {rid : np.where (original_ids == rid) [0] [0] for rid in required_ids}

            # 提取坐标
            left_shoulder = kpts [idx_map [5]]
            right_shoulder = kpts [idx_map [6]]
            left_hip = kpts [idx_map [11]]
            right_hip = kpts [idx_map [12]]

            # 计算特征
            shoulder_width = np.linalg.norm (left_shoulder - right_shoulder)
            hip_width = np.linalg.norm (left_hip - right_hip)
            torso_height = ((left_hip [1] + right_hip [1]) / 2
                            - (left_shoulder [1] + right_shoulder [1]) / 2)

            # 安全校验
            if hip_width < epsilon or abs (torso_height) < epsilon :
                print (f"无效特征值: hip_width={hip_width:.2f}, torso_height={torso_height:.2f}")
                return None

            return [
                shoulder_width / hip_width,
                shoulder_width / abs (torso_height),
                (left_shoulder [0] - left_hip [0]) / hip_width,
                (right_shoulder [0] - right_hip [0]) / hip_width
            ]
        except Exception as e :
            print (f"特征提取失败: {str (e)}")
            return None

    def _assign_body_type (self, features) :
        """体型分类逻辑"""
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

    def _get_raw_keypoints (self, img_id) :
        """获取完整的关键点数据（包含所有17个点）"""
        ann_ids = self.coco.getAnnIds (imgIds=img_id)
        anns = self.coco.loadAnns (ann_ids)
        valid_anns = [a for a in anns if a ['iscrowd'] == 0]
        if not valid_anns :
            return np.zeros ((17, 3))  # 返回空数据

        main_ann = max (valid_anns, key=lambda x : x ['area'])
        kpts = np.array (main_ann ['keypoints']).reshape (-1, 3)

        # 补齐到17个关键点
        if kpts.shape [0] < 17 :
            padded = np.zeros ((17, 3))
            padded [:kpts.shape [0]] = kpts
            return padded
        return kpts


# 分类模型
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


# 可视化工具
# 修改后的可视化方法
class Visualizer :
    @staticmethod
    def visualize_keypoints (dataset, num_samples=5) :
        """关键点可视化（修正索引错误）"""
        plt.figure (figsize=(20, 10))

        for i in range (num_samples) :
            sample = dataset [np.random.choice (len (dataset))]
            if sample is None :
                continue

            features, label = sample
            img_id = dataset.img_ids [i]
            img_info = dataset.coco.loadImgs (img_id) [0]
            img_path = os.path.join (Config.img_dir, img_info ['file_name'])
            img = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)

            # 获取完整的关键点数据（包含所有17个点）
            kpts = dataset._get_raw_keypoints (img_id)

            plt.subplot (2, num_samples, i + 1)
            plt.imshow (img)

            # 绘制所有关键点（按COCO标准索引）
            for kpt_id in range (17) :  # COCO有17个标准关键点
                x, y, v = kpts [kpt_id]
                if v > 0 :
                    # 根据标准ID设置颜色
                    if kpt_id in [5, 6] :
                        color = 'red'  # 肩部
                    elif kpt_id in [11, 12] :
                        color = 'blue'  # 髋部
                    else :
                        color = 'green'  # 其他部位
                    plt.scatter (x, y, c=color, s=50, alpha=0.7)

            plt.axis ('off')
            plt.title (f'Image {i + 1}')

            # 特征可视化（保持不变）
            plt.subplot (2, num_samples, i + num_samples + 1)
            plt.bar (range (4), features.numpy ())
            plt.xticks (range (4), ['S/H Ratio', 'S/Height', 'Left Offset', 'Right Offset'])
            plt.title (f'Label: {label.item ()}')
            plt.ylim (0, 3)

        plt.tight_layout ()
        plt.show ()


# 主流程
def main () :
    # 初始化配置
    try :
        config = Config ()  # 触发路径验证
    except FileNotFoundError as e :
        print (f"配置错误: {str (e)}")
        return

    # 加载数据
    coco = COCO (config.ann_file)
    img_ids = coco.getImgIds (catIds=coco.getCatIds (catNms=['person'])) [:2000]  # 限制样本量

    # 创建数据集
    dataset = BodyDataset (coco, img_ids)

    # 可视化验证
    Visualizer.visualize_keypoints (dataset)
    Visualizer.analyze_features (dataset)

    # 训练模型（可选）
    # train()  # 如果需要训练可以取消注释


if __name__ == "__main__" :
    main ()