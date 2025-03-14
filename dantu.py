# -*- coding: utf-8 -*-
"""
COCO人体体型分类预测系统 - 增强版
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
        # 初始化配置
        self.config = {
            'device' : torch.device ("cuda" if torch.cuda.is_available () else "cpu"),
            'required_kpt_indices' : [5, 6, 11, 12],  # COCO关键点索引
            'class_names' : ['倒三角型', '苹果型', '不对称型', '标准型'],
            'min_keypoints' : 10,
            'min_area' : 5000,
            'max_attempts' : 50
        }

        # 初始化COCO API
        self.coco = COCO (os.path.join (coco_path, "annotations/person_keypoints_train2017.json"))
        self.img_dir = os.path.join (coco_path, "train2017")

        # 预加载有效图片ID
        self.valid_img_ids = self._preload_valid_images ()

        # 加载模型
        self.model = BodyClassifier ().to (self.config ['device'])
        self.model.load_state_dict (torch.load (model_path))
        self.model.eval ()

    def _preload_valid_images (self) :
        """预加载包含有效人体的图片ID"""
        print ("正在扫描数据集...")
        all_ids = self.coco.getImgIds (catIds=self.coco.getCatIds (catNms=['person']))
        valid_ids = []

        for img_id in tqdm (all_ids, desc="验证图片") :
            ann_ids = self.coco.getAnnIds (imgIds=img_id)
            anns = self.coco.loadAnns (ann_ids)

            if self._has_valid_human (anns) :
                valid_ids.append (img_id)

        print (f"找到 {len (valid_ids)} 张有效图片")
        return valid_ids

    def _has_valid_human (self, anns) :
        """验证是否包含有效人体标注"""
        for ann in anns :
            if (ann ['iscrowd'] == 0 and
                    ann ['num_keypoints'] >= self.config ['min_keypoints'] and
                    ann ['area'] > self.config ['min_area']) :
                kpts = np.array (ann ['keypoints']).reshape (-1, 3)
                visible = kpts [:, 2] > 0
                required = set (self.config ['required_kpt_indices'])
                visible_indices = set (np.where (visible) [0].tolist ())
                if required.issubset (visible_indices) :
                    return True
        return False

    def predict_with_retry (self) :
        """带自动重试的预测流程"""
        if not self.valid_img_ids :
            return None, "没有找到有效图片"

        for attempt in range (self.config ['max_attempts']) :
            img_id = np.random.choice (self.valid_img_ids)
            result, msg = self._predict_single (img_id)

            if result is not None :
                print (f"第 {attempt + 1} 次尝试成功")
                return result, msg

        return None, f"超过最大尝试次数 ({self.config ['max_attempts']})"

    def _predict_single (self, img_id) :
        try :
            print (f"\n正在处理图片ID: {img_id}")

            # 加载图像
            img_info = self.coco.loadImgs (img_id) [0]
            print (f"图像尺寸: {img_info ['width']}x{img_info ['height']}")

            # 处理标注
            anns = self.coco.loadAnns (self.coco.getAnnIds (imgIds=img_id))
            print (f"找到{len (anns)}个标注")

            main_ann = self._get_main_annotation (anns)
            print (f"主标注面积: {main_ann ['area']} 关键点数: {main_ann ['num_keypoints']}")

            # 关键点提取
            kpts = self._extract_keypoints (main_ann)
            print ("关键点坐标:\n", kpts)

            # 特征计算
            features = self._calculate_features (kpts)
            print ("计算特征:", features)

            # 模型预测
            prediction = self._model_predict (features)
            return self._visualize (...), prediction

        except Exception as e :
            print (f"失败原因: {str (e)}")
            return None, str (e)

    def _get_main_annotation (self, anns) :
        """选择主人体标注"""
        valid_anns = [a for a in anns if
                      a ['iscrowd'] == 0 and
                      a ['num_keypoints'] >= self.config ['min_keypoints'] and
                      a ['area'] > self.config ['min_area']]
        return max (valid_anns, key=lambda x : x ['area']) if valid_anns else None

    def _extract_keypoints (self, ann) :
        """关键点提取"""
        kpts = np.array (ann ['keypoints'], dtype=np.float32).reshape (-1, 3)
        required = self.config ['required_kpt_indices']
        return np.array ([kpts [i] for i in required if kpts [i, 2] > 0])

    def _calculate_features (self, kpts) :
        """特征计算"""
        # 解包关键点坐标 (左肩, 右肩, 左髋, 右髋)
        ls, rs, lh, rh = kpts [:, :2]

        # 计算几何特征
        shoulder_width = np.linalg.norm (ls - rs)
        hip_width = np.linalg.norm (lh - rh)
        torso_height = np.mean ([lh [1], rh [1]]) - np.mean ([ls [1], rs [1]])

        # 特征工程
        features = [
            shoulder_width / max (hip_width, 1e-6),
            shoulder_width / max (torso_height, 1e-6),
            (ls [0] - lh [0]) / max (hip_width, 1e-6),
            (rs [0] - rh [0]) / max (hip_width, 1e-6)
        ]
        return features

    def _model_predict (self, features) :
        """模型预测"""
        with torch.no_grad () :
            tensor = torch.FloatTensor (features).to (self.config ['device'])
            outputs = self.model (tensor.unsqueeze (0))
            return self.config ['class_names'] [outputs.argmax ().item ()]

    def _visualize (self, img, kpts, prediction) :
        """可视化结果"""
        # 绘制关键点
        colors = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (255, 0, 0)]  # 肩(绿), 髋(红)
        for pt, color in zip (kpts, colors) :
            x, y = map (int, pt [:2])
            cv2.circle (img, (x, y), 8, color, -1)

        # 绘制连接线
        cv2.line (img, tuple (map (int, kpts [0])), tuple (map (int, kpts [1])), (0, 255, 0), 2)  # 双肩
        cv2.line (img, tuple (map (int, kpts [2])), tuple (map (int, kpts [3])), (255, 0, 0), 2)  # 双髋
        cv2.line (img,
                  tuple (map (int, (kpts [0] + kpts [1]) / 2)),
                  tuple (map (int, (kpts [2] + kpts [3]) / 2)),
                  (255, 255, 0), 2)  # 躯干中线

        # 添加预测结果文字
        cv2.putText (img, f"体型: {prediction}", (20, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return cv2.cvtColor (img, cv2.COLOR_RGB2BGR)

    def verify_model (self) :
        test_input = torch.randn (1, 4).to (self.config ['device'])
        try :
            output = self.model (test_input)
            print (f"模型验证通过，输出形状: {output.shape}")
            return True
        except Exception as e :
            print (f"模型验证失败: {str (e)}")
            return False

if __name__ == "__main__" :
    # 初始化预测系统
    predictor = BodyPredictor (
        model_path="body_classifier.pth",
        coco_path=r"C:\datasets\coco2017"
    )
    if predictor.verify_model():
        result_img, prediction = predictor.predict_with_retry()
    # 执行预测
    result_img, prediction = predictor.predict_with_retry ()

    if result_img is not None :
        # 显示并保存结果
        cv2.imshow ("体型分类结果", result_img)
        cv2.waitKey (0)
        cv2.destroyAllWindows ()
        cv2.imwrite ("prediction_result.jpg", result_img)
        print (f"预测成功: {prediction}")
    else :
        print (f"预测失败: {prediction}")