import os
import numpy as np
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# 配置参数
data_path = r"C:\datasets\coco2017"  # 修改为实际路径
ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
img_dir = os.path.join (data_path, "train2017")

# COCO关键点连接配置（包含全部17个关键点）
skeleton = [
    # 身体主干
    (5, 6, 'yellow'),  # 左右肩
    (5, 11, 'green'),  # 左肩-左髋
    (6, 12, 'red'),  # 右肩-右髋
    (11, 12, 'blue'),  # 左右髋

    # 左臂
    (5, 7, 'green'),  # 左肩-左肘
    (7, 9, 'lime'),  # 左肘-左手腕

    # 右臂
    (6, 8, 'red'),  # 右肩-右肘
    (8, 10, 'orange'),  # 右肘-右手腕

    # 左腿
    (11, 13, 'green'),  # 左髋-左膝
    (13, 15, 'lime'),  # 左膝-左脚踝

    # 右腿
    (12, 14, 'red'),  # 右髋-右膝
    (14, 16, 'orange'),  # 右膝-右脚踝

    # 面部
    (0, 1, 'purple'),  # 鼻子-左眼
    (0, 2, 'purple'),  # 鼻子-右眼
    (1, 3, 'purple'),  # 左眼-左耳
    (2, 4, 'purple')  # 右眼-右耳
]

color_map = {
    'green' : (0, 255, 0),  # BGR格式
    'red' : (0, 0, 255),
    'blue' : (255, 0, 0),
    'yellow' : (0, 255, 255),
    'lime' : (0, 255, 0),
    'orange' : (0, 165, 255),
    'purple' : (255, 0, 255)
}


def load_coco_data () :
    """加载COCO数据并筛选有效人体实例"""
    coco = COCO (ann_file)
    cat_ids = coco.getCatIds (catNms=['person'])
    img_ids = coco.getImgIds (catIds=cat_ids)

    valid_entries = []
    for img_id in img_ids :
        ann_ids = coco.getAnnIds (imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns (ann_ids)

        for ann in anns :
            # 筛选条件：非人群标注且至少包含1个关键点
            if ann ['iscrowd'] == 0 and ann ['num_keypoints'] >= 10 :
                valid_entries.append ({
                    'img_id' : img_id,
                    'ann' : ann,
                    'bbox' : ann ['bbox'],
                    'keypoints' : np.array (ann ['keypoints']).reshape (-1, 3)
                })
                break  # 每张图只取一个实例
    return coco, valid_entries


def visualize_full_body (coco, entries, num_samples=9) :
    """可视化完整人体关键点"""
    plt.figure (figsize=(16, 16))

    # 随机采样并排序
    sampled_entries = sorted (np.random.choice (entries, min (num_samples, len (entries)), replace=False),
                              key=lambda x : x ['img_id'])

    for i, entry in enumerate (sampled_entries) :
        img_info = coco.loadImgs (entry ['img_id']) [0]
        img_path = os.path.join (img_dir, img_info ['file_name'])

        # 读取图像
        img = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)
        kpts = entry ['keypoints']

        # 绘制边界框
        bbox = entry ['bbox']
        cv2.rectangle (img,
                       (int (bbox [0]), int (bbox [1])),
                       (int (bbox [0] + bbox [2]), int (bbox [1] + bbox [3])),
                       (255, 165, 0), 2)  # 橙色边框

        # 绘制所有可见关键点
        for kid in range (17) :
            x, y, v = kpts [kid]
            if v > 0 :
                # 根据身体部位设置颜色
                if kid in [5, 7, 9, 11, 13, 15] :  # 左半身
                    color = (0, 255, 0)  # 绿色
                elif kid in [6, 8, 10, 12, 14, 16] :  # 右半身
                    color = (0, 0, 255)  # 红色
                else :  # 面部
                    color = (255, 0, 255)  # 紫色
                cv2.circle (img, (int (x), int (y)), 6, color, -1)

        # 绘制骨架连接线
        for (s, e, color_name) in skeleton :
            if s < 17 and e < 17 :
                x1, y1, v1 = kpts [s]
                x2, y2, v2 = kpts [e]
                if v1 > 0 and v2 > 0 :
                    cv2.line (img,
                              (int (x1), int (y1)),
                              (int (x2), int (y2)),
                              color_map [color_name], 2)

        # 显示图像信息
        plt.subplot (3, 3, i + 1)
        plt.imshow (img)
        plt.title (f"ID:{entry ['img_id']}\nKeypoints:{sum (kpts [:, 2] > 0)}/17")
        plt.axis ('off')

    plt.tight_layout ()
    plt.show ()


if __name__ == "__main__" :
    # 加载数据
    coco, valid_entries = load_coco_data ()
    print (f"找到 {len (valid_entries)} 个有效人体实例")

    # 可视化展示
    if valid_entries :
        visualize_full_body (coco, valid_entries)
    else :
        print ("未找到符合要求的样本")