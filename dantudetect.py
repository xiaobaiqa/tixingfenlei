import os
import random
import numpy as np
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


class BodyTypeAnalyzer :
    def __init__ (self, data_path) :
        self.data_path = data_path
        self.ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
        self.img_dir = os.path.join (data_path, "train2017")
        self.coco = COCO (self.ann_file)
        self.skeleton = [
            (5, 6, 'yellow'), (5, 7, 'green'), (7, 9, 'lime'),
            (6, 8, 'red'), (8, 10, 'orange'), (5, 11, 'green'),
            (6, 12, 'red'), (11, 12, 'blue'), (11, 13, 'green'),
            (13, 15, 'lime'), (12, 14, 'red'), (14, 16, 'orange')
        ]
        self.color_map = {
            'green' : (0, 255, 0), 'red' : (0, 0, 255),
            'blue' : (255, 0, 0), 'yellow' : (0, 255, 255),
            'lime' : (0, 255, 0), 'orange' : (0, 165, 255)
        }

    def load_valid_instances (self, min_keypoints=10) :
        """加载所有有效实例"""
        cat_ids = self.coco.getCatIds (catNms=['person'])
        img_ids = self.coco.getImgIds (catIds=cat_ids)

        valid_entries = []
        for img_id in img_ids :
            ann_ids = self.coco.getAnnIds (imgIds=img_id, catIds=cat_ids, iscrowd=False)
            anns = self.coco.loadAnns (ann_ids)

            for ann in anns :
                if ann ['num_keypoints'] >= min_keypoints :
                    kpts = np.array (ann ['keypoints']).reshape (-1, 3)
                    if self._validate_keypoints (kpts) :
                        valid_entries.append ({
                            'img_id' : img_id,
                            'ann' : ann,
                            'kpts' : kpts,
                            'bbox' : ann ['bbox']
                        })
                        break  # 每图取一个实例
        return valid_entries

    def _validate_keypoints (self, kpts) :
        """验证必要关键点可见性"""
        required = [5, 6, 11, 12]  # 双肩和双髋
        return all (kpts [i, 2] > 0 for i in required)

    def classify_body_type (self, kpts) :
        """执行体型分类"""
        ratios = self._calculate_ratios (kpts)
        return self._classify (ratios), ratios

    def _calculate_ratios (self, kpts) :
        """计算关键比例"""
        ls = kpts [5] [:2]
        rs = kpts [6] [:2]
        lh = kpts [11] [:2]
        rh = kpts [12] [:2]

        shoulder_width = np.linalg.norm (ls - rs)
        hip_width = np.linalg.norm (lh - rh)
        torso_height = np.mean ([lh [1], rh [1]]) - np.mean ([ls [1], rs [1]])

        eps = 1e-6
        return {
            'shoulder_hip_ratio' : shoulder_width / (hip_width + eps),
            'waist_hip_ratio' : (0.7 * shoulder_width) / (hip_width + eps),
            'torso_proportion' : torso_height / (hip_width + eps)
        }

    def _classify (self, ratios) :
        """分类逻辑"""
        shr = ratios ['shoulder_hip_ratio']
        whr = ratios ['waist_hip_ratio']

        if shr > 1.15 and whr < 0.85 :
            return 'Hourglass type'
        elif shr > 1.1 and whr >= 0.9 :
            return 'Inverted triangle'
        elif shr < 0.95 and whr < 0.98 :
            return 'Pear type'
        elif 0.98 <= shr <= 1.05 and whr > 0.95 :
            return 'Apple-type'
        else :
            return 'Standard'

    def visualize_results (self, entries, output_dir='output') :
        """批量可视化结果"""
        os.makedirs (output_dir, exist_ok=True)
        plt.figure (figsize=(15, 10))

        for i, entry in enumerate (entries) :
            # 加载图像
            img_info = self.coco.loadImgs (entry ['img_id']) [0]
            img_path = os.path.join (self.img_dir, img_info ['file_name'])
            img = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)

            # 绘制关键点和骨架
            self._draw_pose (img, entry ['kpts'])

            # 添加分类信息
            body_type, ratios = entry ['result']
            text = f"{body_type}\nSHR:{ratios ['shoulder_hip_ratio']:.2f} WHR:{ratios ['waist_hip_ratio']:.2f}"

            # 显示子图
            plt.subplot (2, 3, i + 1)
            plt.imshow (img)
            plt.title (f"Image ID: {entry ['img_id']}\n{text}")
            plt.axis ('off')

        plt.tight_layout ()

        # 保存并显示
        output_path = os.path.join (output_dir, "random_results.jpg")
        plt.savefig (output_path, bbox_inches='tight')
        plt.show ()
        return output_path

    def _draw_pose (self, img, kpts) :
        """绘制骨骼关键点"""
        # 绘制关键点
        for kid in range (17) :
            x, y, v = kpts [kid]
            if v > 0 :
                color = (0, 255, 0) if kid in [5, 7, 9, 11, 13, 15] else (0, 0, 255)
                cv2.circle (img, (int (x), int (y)), 5, color, -1)

        # 绘制骨架
        for (s, e, color_name) in self.skeleton :
            if s < 17 and e < 17 :
                x1, y1, v1 = kpts [s]
                x2, y2, v2 = kpts [e]
                if v1 > 0 and v2 > 0 :
                    cv2.line (img, (int (x1), int (y1)), (int (x2), int (y2)),
                              self.color_map [color_name], 2)


if __name__ == "__main__" :
    # 初始化分析器
    analyzer = BodyTypeAnalyzer (r"C:\datasets\coco2017")

    # 加载所有有效实例
    all_instances = analyzer.load_valid_instances (min_keypoints=10)

    if not all_instances :
        print ("未找到有效样本")
        exit ()

    # 随机选取5个样本（允许重复）
    random.seed ()  # 使用系统时间作为随机种子
    selected_instances = random.sample (all_instances, k=min (5, len (all_instances)))

    # 进行分类
    for entry in selected_instances :
        body_type, ratios = analyzer.classify_body_type (entry ['kpts'])
        entry ['result'] = (body_type, ratios)

    # 批量可视化
    output_path = analyzer.visualize_results (selected_instances)
    print (f"随机抽样结果已保存至：{output_path}")