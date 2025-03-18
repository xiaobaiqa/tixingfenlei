import json
import os
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2


def load_and_validate_metadata (metadata_path) :
    """加载并验证元数据"""
    valid_data = []
    invalid_data = []

    # 加载原始数据
    with open (metadata_path, 'r') as f :
        try :
            metadata = json.load (f)
        except json.JSONDecodeError :
            raise ValueError ("元数据文件格式错误，请检查JSON格式")

    # 数据验证
    for item in tqdm (metadata, desc="验证数据") :
        # 检查必要字段
        if 'img_path' not in item or 'label' not in item :
            invalid_data.append (item)
            continue

        # 验证路径有效性
        img_path = item ['img_path']
        if not os.path.exists (img_path) :
            invalid_data.append (item)
            continue

        # 验证标签有效性
        if not isinstance (item ['label'], int) or item ['label'] < 0 or item ['label'] > 4 :
            invalid_data.append (item)
            continue

        valid_data.append (item)

    return valid_data, invalid_data


def analyze_dataset (metadata) :
    """生成数据集分析报告"""
    analysis = {
        'total_samples' : len (metadata),
        'class_distribution' : defaultdict (int),
        'image_size_distribution' : defaultdict (int),
        'random_samples' : []
    }

    # 类别分布统计
    for item in metadata :
        analysis ['class_distribution'] [item ['label']] += 1

    # 图像尺寸抽样检查（检查前50个样本）
    for item in metadata [:50] :
        try :
            img = cv2.imread (item ['img_path'])
            h, w = img.shape [:2]
            analysis ['image_size_distribution'] [(w, h)] += 1
        except :
            analysis ['image_size_distribution'] ['error'] += 1

    # 随机样本展示
    analysis ['random_samples'] = np.random.choice (
        [os.path.basename (item ['img_path']) for item in metadata],
        size=5,
        replace=False
    ).tolist ()

    return analysis


def split_dataset (metadata, test_size=0.2, random_state=42) :
    """分层分割数据集"""
    labels = [item ['label'] for item in metadata]

    # 保持类别分布的分割
    train_data, val_data = train_test_split (
        metadata,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # 验证分割后的分布
    train_labels = [x ['label'] for x in train_data]
    val_labels = [x ['label'] for x in val_data]

    print ("\n分割结果验证:")
    print (f"训练集类别分布: {dict (zip (*np.unique (train_labels, return_counts=True)))}")
    print (f"验证集类别分布: {dict (zip (*np.unique (val_labels, return_counts=True)))}")

    return train_data, val_data


def save_split_data (train_data, val_data, output_dir) :
    """保存分割后的数据"""
    os.makedirs (output_dir, exist_ok=True)

    with open (os.path.join (output_dir, 'train_metadata.json'), 'w') as f :
        json.dump (train_data, f, indent=2)

    with open (os.path.join (output_dir, 'val_metadata.json'), 'w') as f :
        json.dump (val_data, f, indent=2)


if __name__ == "__main__" :
    # 配置参数
    metadata_path = r"D:\biyesheji\code\tixingfenlei\metadata.json"
    output_dir = r"D:\biyesheji\code\tixingfenlei\split_data"

    # 步骤1：加载并验证数据
    valid_data, invalid_data = load_and_validate_metadata (metadata_path)

    print (f"\n数据有效性报告:")
    print (f"总样本量: {len (valid_data) + len (invalid_data)}")
    print (f"有效样本: {len (valid_data)}")
    print (f"无效样本: {len (invalid_data)} (建议检查路径和标签)")

    # 步骤2：分析数据集
    analysis = analyze_dataset (valid_data)

    print ("\n数据集分析:")
    print (f"1. 类别分布:")
    for cls, count in sorted (analysis ['class_distribution'].items ()) :
        print (f"   类别{cls}: {count} 样本 ({count / analysis ['total_samples']:.1%})")

    print (f"\n2. 图像尺寸抽样检查 (前50个样本):")
    for size, count in analysis ['image_size_distribution'].items () :
        print (f"   尺寸{size}: {count} 张")

    print (f"\n3. 随机样本示例:")
    for sample in analysis ['random_samples'] :
        print (f"   - {sample}")

    # 步骤3：分割数据集
    train_data, val_data = split_dataset (valid_data)

    # 步骤4：保存分割结果
    save_split_data (train_data, val_data, output_dir)
    print (f"\n分割数据已保存至: {output_dir}")