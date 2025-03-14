# -*- coding: utf-8 -*-
"""
增强版单图预测系统 - 自动筛选有效样本
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from tqdm import tqdm


# 模型定义
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


class BodyPredictor :
    def __init__ (self, model_path, coco_path) :
        # 加载配置
        self.config = type ('Config', (), {
            'device' : torch.device ("cuda" if torch.cuda.is_available () else "cpu"),
            'required_kpt_indices' : [5, 6, 11, 12],
            'class_names' : ['倒三角型', '苹果型', '不对称型', '标准型'],
            'min_keypoints' : 10,
            'min_area' : 5000
        }) ()

        # 初始化COCO API
        self.coco = COCO (os.path.join (coco_path, "annotations/person_keypoints_train2017.json"))
        self.img_dir = os.path.join (coco_path, "train2017")

        # 预加载有效图片ID
        self.valid_img_ids = self._preload_valid_images ()

        # 加载模型
        self.model = BodyClassifier ().to (self.config.device)
        self.model.load_state_dict (torch.load (model_path))
        self.model.eval ()
        self._attempted_ids = set ()  # 跟踪已尝试ID
        # 关键点颜色映射
        self.kpt_colors = {
            5 : (0, 255, 0),  # 左肩-绿色
            6 : (0, 255, 0),  # 右肩-绿色
            11 : (255, 0, 0),  # 左髋-红色
            12 : (255, 0, 0)  # 右髋-红色
        }

    def _preload_valid_images (self) :
        """预加载包含有效人体的图片ID"""
        print ("Preloading valid image IDs...")
        all_img_ids = self.coco.getImgIds (catIds=self.coco.getCatIds (catNms=['person']))
        valid_ids = []

        for img_id in tqdm (all_img_ids) :
            ann_ids = self.coco.getAnnIds (imgIds=img_id)
            anns = self.coco.loadAnns (ann_ids)

            if self._has_valid_annotation (anns) :
                valid_ids.append (img_id)

        print (f"Found {len (valid_ids)} valid images")
        return valid_ids

    def _has_valid_annotation (self, anns) :
        """验证标注是否有效"""
        for ann in anns :
            if (ann ['iscrowd'] == 0 and
                    ann ['num_keypoints'] >= self.config.min_keypoints and
                    ann ['area'] > self.config.min_area) :

                kpts = np.array (ann ['keypoints']).reshape (-1, 3)
                visible = kpts [:, 2] > 0
                required_indices = set (self.config.required_kpt_indices)
                visible_indices = set (np.where (visible) [0].tolist ())

                if required_indices.issubset (visible_indices) :
                    return True
        return False

    def get_random_valid_image (self) :
        """获取随机有效图片"""
        return np.random.choice (self.valid_img_ids)

    def predict_single_image (self, img_id) :
        """增强版预测流程"""
        try :
            # 加载图像元数据
            img_info = self.coco.loadImgs (img_id) [0]
            img_path = os.path.join (self.img_dir, img_info ['file_name'])

            # 加载图像
            img = cv2.imread (img_path)
            if img is None :
                return None, "Image loading failed"
            img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)

            # 获取标注信息
            ann_ids = self.coco.getAnnIds (imgIds=img_id)
            anns = self.coco.loadAnns (ann_ids)

            # 选择主人体实例
            main_ann = self._select_main_annotation (anns)
            if main_ann is None :
                return None, "No valid human detected"

            # 提取关键点
            kpts = self._extract_keypoints (main_ann)
            if kpts is None :
                return None, "Keypoints extraction failed"

            # 计算特征
            features = self._calculate_features (kpts)
            if features is None :
                return None, "Feature calculation failed"

            # 执行预测
            prediction = self._model_predict (features)

            # 可视化结果
            vis_img = self._visualize (img.copy (), kpts, prediction)
            return vis_img, prediction

        except Exception as e :
            return None, f"Prediction failed: {str (e)}"

    def _select_main_annotation (self, anns) :
        """选择最大有效标注"""
        valid_anns = []
        for ann in anns :
            if (ann ['iscrowd'] == 0 and
                    ann ['num_keypoints'] >= self.config.min_keypoints and
                    ann ['area'] > self.config.min_area) :
                valid_anns.append (ann)

        if not valid_anns :
            return None

        return max (valid_anns, key=lambda x : x ['area'])

    def _extract_keypoints (self, ann) :
        """增强版关键点提取"""
        kpts = np.array (ann ['keypoints'], dtype=np.float32).reshape (-1, 3)

        # 验证关键点索引范围
        max_index = max (self.config.required_kpt_indices)
        if len (kpts) <= max_index :
            return None

        # 提取可见关键点
        selected = []
        for idx in self.config.required_kpt_indices :
            if kpts [idx, 2] == 0 :  # 关键点不可见
                return None
            selected.append (kpts [idx, :2])

        return np.array (selected)

    def _calculate_features (self, kpts) :
        """增强版特征计算"""
        try :
            left_shoulder = kpts [0]
            right_shoulder = kpts [1]
            left_hip = kpts [2]
            right_hip = kpts [3]

            # 计算几何特征
            shoulder_width = np.linalg.norm (left_shoulder - right_shoulder)
            hip_width = np.linalg.norm (left_hip - right_hip)
            torso_height = np.mean ([left_hip [1], right_hip [1]]) - np.mean ([left_shoulder [1], right_shoulder [1]])

            # 有效性检查
            if hip_width < 1e-6 or torso_height < 1e-6 :
                return None

            features = [
                shoulder_width / hip_width,
                shoulder_width / torso_height,
                (left_shoulder [0] - left_hip [0]) / hip_width,
                (right_shoulder [0] - right_hip [0]) / hip_width
            ]

            # 检查特征有效性
            if any (np.isnan (f) or np.isinf (f) for f in features) :
                return None

            return features
        except :
            return None

    def _model_predict (self, features) :
        """执行模型预测"""
        with torch.no_grad () :
            tensor = torch.FloatTensor (features).to (self.config.device)
            output = self.model (tensor.unsqueeze (0))
            pred_class = output.argmax ().item ()
        return self.config.class_names [pred_class]

    def _visualize (self, img, kpts, prediction) :
        """可视化关键点和预测结果"""
        # 绘制关键点
        for idx, pt in zip (self.config.required_kpt_indices, kpts) :
            x, y = map (int, pt)
            cv2.circle (img, (x, y), 8, self.kpt_colors [idx], -1)

        # 绘制连接线
        cv2.line (img, tuple (map (int, kpts [0])), tuple (map (int, kpts [1])), (0, 255, 0), 2)  # 双肩
        cv2.line (img, tuple (map (int, kpts [2])), tuple (map (int, kpts [3])), (255, 0, 0), 2)  # 双髋
        cv2.line (img,
                  tuple (map (int, (kpts [0] + kpts [1]) / 2)),
                  tuple (map (int, (kpts [2] + kpts [3]) / 2)),
                  (255, 255, 0), 2)  # 躯干中线

        # 添加文字标注
        text = f"Prediction: {prediction}"
        cv2.putText (img, text, (20, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return cv2.cvtColor (img, cv2.COLOR_RGB2BGR)

    def predict_until_success (self, max_attempts=100, show_progress=True) :
        """增强版持续预测方法"""
        if not self.valid_img_ids :
            return None, "No valid images available"

        attempt = 0
        pbar = tqdm (total=max_attempts, desc="Predicting", disable=not show_progress)

        while attempt < max_attempts :
            # 确保不重复尝试相同ID
            if len (self._attempted_ids) >= len (self.valid_img_ids) :
                self._attempted_ids.clear ()

            while True :
                img_id = np.random.choice (self.valid_img_ids)
                if img_id not in self._attempted_ids :
                    self._attempted_ids.add (img_id)
                    break

            result_img, prediction = self.predict_single_image (img_id)

            if result_img is not None :
                pbar.close ()
                return result_img, prediction

            attempt += 1
            pbar.update (1)
            pbar.set_postfix ({"attempt" : attempt, "status" : "retrying"})

        pbar.close ()
        return None, f"Failed after {max_attempts} attempts"


# 使用示例
if __name__ == "__main__" :
    # 初始化预测器
    predictor = BodyPredictor (
        model_path="body_classifier.pth",
        coco_path=r"C:\datasets\coco2017"  # 修改为实际路径
    )

    # 获取并预测随机有效图片
    if not predictor.valid_img_ids :
        print ("没有找到有效图片，请检查数据集路径和标注文件")
    else :
        sample_id = predictor.get_random_valid_image ()
        result_img, prediction = predictor.predict_single_image (sample_id)

        if result_img is not None :
            cv2.imshow ("Prediction Result", result_img)
            cv2.waitKey (0)
            cv2.destroyAllWindows ()
            print (f"预测成功: {prediction}")
        else :
            print (f"预测失败: {prediction}")