import os
import json
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

plt.rcParams ['font.sans-serif'] = ['SimHei']
plt.rcParams ['axes.unicode_minus'] = False
# 配置参数
class Config :
    metadata_path = r"D:\biyesheji\code\tixingfenlei\metadata.json"
    output_dir = r"D:\biyesheji\code\tixingfenlei\split_data"
    class_names = {
        0 : "倒三角形",
        1 : "矩形(H型)",
        2 : "椭圆形"
    }


def visualize_sample (item, show_keypoints=True) :
    """可视化样本图像及特征点"""
    plt.figure (figsize=(12, 6))

    # 读取原始图像
    img = cv2.imread (item ['img_path'])
    if img is None :
        print (f"无法读取图像: {item ['img_path']}")
        return

    img_rgb = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
    h, w = img.shape [:2]

    # 创建子图画布
    ax1 = plt.subplot (1, 2, 1)
    ax2 = plt.subplot (1, 2, 2)

    # 显示原始图像
    ax1.imshow (img_rgb)
    ax1.set_title (f"原始图像\n{os.path.basename (item ['img_path'])}\n尺寸: {w}x{h}")
    ax1.axis ('off')

    # 显示带特征点的图像
    annotated_img = img_rgb.copy ()
    if show_keypoints and 'keypoints' in item :
        try :
            kpts = np.array (item ['keypoints']).reshape (-1, 3)

            # 绘制关键点骨架连接
            connections = [
                (5, 6), (5, 11), (6, 12),  # 躯干
                (5, 7), (7, 9),  # 左臂
                (6, 8), (8, 10),  # 右臂
                (11, 13), (13, 15),  # 左腿
                (12, 14), (14, 16)  # 右腿
            ]

            # 绘制关键点
            for idx in [5, 6, 11, 12] :  # 只显示关键躯干点
                x, y, v = kpts [idx]
                if v > 0 :
                    color = 'red' if idx % 2 == 0 else 'blue'
                    ax2.scatter (x, y, c=color, s=50, edgecolors='white')
                    ax2.text (x + 10, y + 10, f"{idx}", color='white', fontsize=8,
                              bbox=dict (facecolor=color, alpha=0.7, pad=1))

            # 计算特征参数
            ls = kpts [5] [:2]
            rs = kpts [6] [:2]
            lh = kpts [11] [:2]
            rh = kpts [12] [:2]

            shoulder_width = np.linalg.norm (ls - rs)
            hip_width = np.linalg.norm (lh - rh)
            waist_width = 0.75 * (shoulder_width + hip_width) / 2
            shoulder_hip_ratio = shoulder_width / (hip_width + 1e-6)
            waist_hip_ratio = waist_width / (hip_width + 1e-6)

            # 添加特征参数标注
            feature_text = f"肩宽: {shoulder_width:.1f}\n臀宽: {hip_width:.1f}\n"
            feature_text += f"肩臀比: {shoulder_hip_ratio:.2f}\n腰臀比: {waist_hip_ratio:.2f}"
            ax2.text (10, 30, feature_text,
                      color='white', fontsize=9,
                      bbox=dict (facecolor='black', alpha=0.5))

            # 绘制躯干中心线
            shoulder_center = ((ls [0] + rs [0]) / 2, (ls [1] + rs [1]) / 2)
            hip_center = ((lh [0] + rh [0]) / 2, (lh [1] + rh [1]) / 2)
            con = ConnectionPatch (shoulder_center, hip_center,
                                   coordsA="data", coordsB="data",
                                   arrowstyle="-|>", shrinkA=5, shrinkB=5,
                                   mutation_scale=20, fc="cyan")
            ax2.add_artist (con)
        except Exception as e :
            print (f"特征点处理失败: {str (e)}")

    ax2.imshow (annotated_img)
    ax2.set_title (f"特征分析\n真实标签: {Config.class_names.get (item ['label'], '未知')}")
    ax2.axis ('off')

    plt.tight_layout ()
    plt.show ()


def load_and_validate_metadata (metadata_path) :
    """加载并验证元数据"""
    valid_data = []
    invalid_data = []

    with open (metadata_path, 'r') as f :
        try :
            metadata = json.load (f)
        except json.JSONDecodeError :
            raise ValueError ("元数据文件格式错误")

    for item in tqdm (metadata, desc="验证数据") :
        # 验证必要字段
        if 'img_path' not in item or 'label' not in item :
            invalid_data.append (item)
            continue

        # 验证图像路径
        if not os.path.exists (item ['img_path']) :
            invalid_data.append (item)
            continue

        # 验证标签有效性
        if item ['label'] not in Config.class_names :
            invalid_data.append (item)
            continue

        valid_data.append (item)

    return valid_data, invalid_data


def analyze_dataset (metadata, show_samples=3) :
    """增强版数据集分析"""
    analysis = {
        'total_samples' : len (metadata),
        'class_distribution' : defaultdict (int),
        'feature_distribution' : defaultdict (list),
        'random_samples' : []
    }

    # 随机可视化样本
    print ("\n随机样本可视化：")
    samples_to_show = np.random.choice (metadata, size=min (show_samples, len (metadata)), replace=False)
    for item in samples_to_show :
        visualize_sample (item)

        # 收集特征数据
        try :
            kpts = np.array (item ['keypoints']).reshape (-1, 3)
            ls = kpts [5] [:2]
            rs = kpts [6] [:2]
            lh = kpts [11] [:2]
            rh = kpts [12] [:2]

            shoulder_width = np.linalg.norm (ls - rs)
            hip_width = np.linalg.norm (lh - rh)
            waist_width = 0.75 * (shoulder_width + hip_width) / 2

            analysis ['feature_distribution'] ['shoulder_hip_ratio'].append (shoulder_width / (hip_width + 1e-6))
            analysis ['feature_distribution'] ['waist_hip_ratio'].append (waist_width / (hip_width + 1e-6))
        except Exception as e :
            print (f"特征提取失败: {str (e)}")

    # 绘制特征分布
    plt.figure (figsize=(12, 5))

    plt.subplot (121)
    plt.hist (analysis ['feature_distribution'] ['shoulder_hip_ratio'], bins=20, color='blue', alpha=0.7)
    plt.title ('肩臀比分布')
    plt.xlabel ('比例')
    plt.ylabel ('样本数')

    plt.subplot (122)
    plt.hist (analysis ['feature_distribution'] ['waist_hip_ratio'], bins=20, color='green', alpha=0.7)
    plt.title ('腰臀比分布')
    plt.xlabel ('比例')

    plt.tight_layout ()
    plt.show ()

    # 统计类别分布
    for item in metadata :
        analysis ['class_distribution'] [item ['label']] += 1

    return analysis


def split_and_save_data (valid_data) :
    """数据分割与保存"""
    # 分层分割
    labels = [item ['label'] for item in valid_data]
    train_data, val_data = train_test_split (
        valid_data,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # 保存结果
    os.makedirs (Config.output_dir, exist_ok=True)
    with open (os.path.join (Config.output_dir, 'train_metadata.json'), 'w') as f :
        json.dump (train_data, f, indent=2)
    with open (os.path.join (Config.output_dir, 'val_metadata.json'), 'w') as f :
        json.dump (val_data, f, indent=2)

    return train_data, val_data


if __name__ == "__main__" :
    # 1. 加载数据
    valid_data, invalid_data = load_and_validate_metadata (Config.metadata_path)

    print (f"\n数据有效性报告:")
    print (f"总样本量: {len (valid_data) + len (invalid_data)}")
    print (f"有效样本: {len (valid_data)}")
    print (f"无效样本: {len (invalid_data)}")

    # 2. 数据分析
    analysis = analyze_dataset (valid_data)

    print ("\n类别分布统计:")
    for cls, count in analysis ['class_distribution'].items () :
        print (f"{Config.class_names [cls]}: {count} 样本 ({count / len (valid_data):.1%})")

    # 3. 分割与保存
    train_data, val_data = split_and_save_data (valid_data)
    print (f"\n数据已分割保存至: {Config.output_dir}")
    print (f"训练集: {len (train_data)} 样本")
    print (f"验证集: {len (val_data)} 样本")