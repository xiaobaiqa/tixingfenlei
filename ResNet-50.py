import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import torchvision.models as models  # 新增导入
from torchvision.models.resnet import ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 更新类别映射
class_names = {
    0: "倒三角形",
    1: "矩形(H型)",
    2: "椭圆形"
}


# 配置参数
class Config :
    data_path = r"C:\datasets\coco2017"
    ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 8  # 可适当增大batch_size
    num_classes = 3
    num_epoch = 10
    img_size = (224, 224)  # 修改为标准输入尺寸
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")


# 图像预处理函数
def preprocess_image (img) :
    img = cv2.resize (img, (Config.img_size [1], Config.img_size [0]))  # 使用新尺寸
    img = img.astype (np.float32) / 255.0
    img = (img - Config.mean) / Config.std
    return img.transpose (2, 0, 1)


# 数据集类（保持不变）
class CocoBodyDataset (Dataset) :
    def __init__ (self, metadata) :
        self.metadata = metadata

    def __len__ (self) :
        return len (self.metadata)

    def __getitem__ (self, idx) :
        item = self.metadata [idx]
        img = cv2.imread (item ["img_path"])
        if img is None :
            return torch.zeros (3, *Config.img_size), torch.tensor (-1)
        img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
        img = preprocess_image (img)
        return torch.FloatTensor (img), torch.tensor (item ["label"], dtype=torch.long)


# 在模型定义部分添加正则化
class CustomResNet (nn.Module) :
    def __init__ (self) :
        super ().__init__ ()
        # 保留原始ResNet50的全部结构
        self.base = models.resnet50 (weights=None)
        # 仅替换最后的全连接层
        self.base.fc = nn.Sequential (
            nn.Dropout (0.7),
            nn.Linear (2048, 512),
            nn.ReLU (),
            nn.Dropout (0.3),
            nn.Linear (512, Config.num_classes)
        )

    def forward (self, x) :
        return self.base (x)


def generate_metadata () :
    if os.path.exists ("metadata.json") :
        with open ("metadata.json") as f :
            return json.load (f)

    coco = COCO (Config.ann_file)
    metadata = []
    class_counts = {cls : 0 for cls in class_names.keys ()}
    target_per_class = 500

    # 增强过滤参数
    HIP_X_DIFF_RATIO = 0.25
    SHOULDER_X_DIFF_RATIO = 0.25
    MIN_VISIBILITY = 0
    MIN_HEIGHT_RATIO = 0.25  # 人物最小高度占比
    MIN_AREA_RATIO = 0.1  # 人物最小面积占比
    MIN_JOINT_DIST = 0.15  # 关键点最小间距（相对图像宽度）

    for img_id in coco.getImgIds (catIds=coco.getCatIds ('person')) :
        if all (count >= target_per_class for count in class_counts.values ()) :
            break

        img_info = coco.loadImgs (img_id) [0]
        img_path = os.path.join (Config.img_dir, img_info ["file_name"])
        img_w, img_h = img_info ["width"], img_info ["height"]
        img_area = img_w * img_h

        if not os.path.exists (img_path) :
            continue

        # 阶段1：筛选有效标注
        valid_anns = []
        for ann in coco.loadAnns (coco.getAnnIds (imgIds=img_id, iscrowd=False)) :
            if ann ["num_keypoints"] < 10 :
                continue

            # 计算边界框参数
            x, y, w, h = ann ["bbox"]
            ann_area = ann ["area"]
            height_ratio = h / img_h
            area_ratio = ann_area / img_area

            # 尺寸过滤
            if (height_ratio < MIN_HEIGHT_RATIO) or (area_ratio < MIN_AREA_RATIO) :
                continue

            valid_anns.append (ann)

        # 选择图像中面积最大的标注
        if not valid_anns :
            continue
        main_ann = max (valid_anns, key=lambda x : x ["area"])

        # 阶段2：关键点质量验证
        kpts = np.array (main_ann ["keypoints"]).reshape (-1, 3)
        required_joints = [5, 6, 11, 12]

        # 可见性检查
        if not all (kpts [i, 2] >= MIN_VISIBILITY for i in required_joints) :
            continue

        # 提取关键点坐标
        ls = kpts [5]
        rs = kpts [6]
        lh = kpts [11]
        rh = kpts [12]

        # 姿势质量检查
        def check_joint_dist (j1, j2) :
            dist = np.linalg.norm (j1 [:2] - j2 [:2])
            return dist > (img_w * MIN_JOINT_DIST)

        # 关键点间距检查
        if not (check_joint_dist (ls, rs) and check_joint_dist (lh, rh)) :
            continue

        # 肩部垂直对齐检查
        if abs (ls [1] - rs [1]) > 0.1 * img_h :  # 允许10%高度差
            continue

        # 躯干朝向检查
        shoulder_center = (ls [0] + rs [0]) / 2
        hip_center = (lh [0] + rh [0]) / 2
        if abs (shoulder_center - hip_center) > 0.1 * img_w :
            continue

        # 性别特征检查
        shoulder_width = np.linalg.norm (ls [:2] - rs [:2])
        hip_width = np.linalg.norm (lh [:2] - rh [:2])
        shoulder_hip_ratio = shoulder_width / (hip_width + 1e-6)
        if shoulder_hip_ratio < 0.9 :
            continue

        # 分类并保存
        label = classify_body_type (kpts)
        if class_counts [label] < target_per_class :
            metadata.append ({
                "img_path" : img_path,
                "label" : label,
                "keypoints" : kpts.tolist (),
                "bbox" : main_ann ["bbox"],
                "img_size" : [img_w, img_h]
            })
            class_counts [label] += 1
            print (f"\r采集进度: { {k : v for k, v in class_counts.items () if v > 0} }", end="")

    # 平衡采样
    balanced_metadata = []
    for cls in class_names :
        cls_samples = [m for m in metadata if m ["label"] == cls] [:target_per_class]
        balanced_metadata.extend (cls_samples)

    with open ("metadata.json", "w") as f :
        json.dump (balanced_metadata, f)

    print (
        f"\n最终样本分布: { {cls : len ([m for m in balanced_metadata if m ['label'] == cls]) for cls in class_names} }")
    return balanced_metadata

# 增强的体型分类规则
def classify_body_type (kpts) :
    # 关键点索引（COCO格式）
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_EAR = 3
    RIGHT_EAR = 4

    # 获取关键点坐标
    ls = kpts [LEFT_SHOULDER] [:2]
    rs = kpts [RIGHT_SHOULDER] [:2]
    lh = kpts [LEFT_HIP] [:2]
    rh = kpts [RIGHT_HIP] [:2]
    le = kpts [LEFT_EAR] [:2]
    re = kpts [RIGHT_EAR] [:2]

    # 计算基准维度
    shoulder_width = np.linalg.norm (ls - rs)
    hip_width = np.linalg.norm (lh - rh)
    eps = 1e-6  # 防止除以零

    # 重新定义特征计算
    chest_center = (le + re) / 2
    hip_center = (lh + rh) / 2
    torso_height = np.linalg.norm (chest_center - hip_center)

    # 关键比例系数
    shoulder_hip_ratio = shoulder_width / (hip_width + eps)
    waist_width = 0.75 * (shoulder_width + hip_width) / 2  # 综合估算腰宽
    waist_hip_ratio = waist_width / (hip_width + eps)

    # 分类逻辑（倒三角形、矩形、椭圆形）
    if shoulder_hip_ratio > 1.15 and waist_hip_ratio < 0.85 :
        return 0  # 倒三角形

    # 放宽椭圆形标准
    elif (waist_hip_ratio > 1.05) or \
            (waist_width > shoulder_width) or \
            (hip_width > shoulder_width * 1.05) :
        return 2  # 椭圆形

    # 矩形判断条件
    elif 0.85 < shoulder_hip_ratio < 1.15 and 0.8 < waist_hip_ratio < 1.1 :
        return 1  # 矩形

    # 默认情况处理
    else :
        # 当肩宽明显大于臀宽时归为倒三角形
        if shoulder_hip_ratio > 1.1 :
            return 0
        # 腰臀特征不明确时优先归为椭圆形
        return 2
# 修改后的训练流程
def train () :
    metadata = generate_metadata ()
    train_meta, val_meta = train_test_split (metadata, test_size=0.2, random_state=42)

    train_set = CocoBodyDataset (train_meta)
    val_set = CocoBodyDataset (val_meta)

    train_loader = DataLoader (
        train_set, batch_size=Config.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader (
        val_set, batch_size=Config.batch_size,
        num_workers=0, collate_fn=collate_fn
    )

    # 初始化改进后的模型
    model = CustomResNet ().to (Config.device)
    # 加载预训练权重（如果需要）
    pretrained = models.resnet50 (weights=ResNet50_Weights.IMAGENET1K_V1)
    model.base.load_state_dict (pretrained.state_dict (), strict=False)
    # 使用更强的权重衰减
    optimizer = torch.optim.AdamW (
        model.parameters (),
        lr=1e-4,
        weight_decay=0.01  # 增大正则化强度
    )

    # 添加标签平滑的损失函数
    class LabelSmoothLoss (nn.Module) :
        def __init__ (self, smoothing=0.1) :
            super ().__init__ ()
            self.smoothing = smoothing

        def forward (self, inputs, targets) :
            log_probs = torch.log_softmax (inputs, dim=-1)
            nll_loss = -log_probs.gather (dim=-1, index=targets.unsqueeze (1))
            nll_loss = nll_loss.squeeze (1)
            smooth_loss = -log_probs.mean (dim=-1)
            loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
            return loss.mean ()

    criterion = LabelSmoothLoss (smoothing=0.1)

    # 早停机制参数
    best_acc = 0.0
    patience = 3
    no_improve = 0
    early_stop = False

    for epoch in range (Config.num_epoch) :  # epoch数
        if early_stop :
            print ("早停触发，停止训练")
            break

        model.train ()
        running_loss = 0.0

        # 训练阶段
        for inputs, labels in train_loader :
            if len (inputs) == 0 :
                continue
            inputs, labels = inputs.to (Config.device), labels.to (Config.device)

            optimizer.zero_grad ()
            outputs = model (inputs)
            loss = criterion (outputs, labels)
            loss.backward ()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_ (model.parameters (), max_norm=1.0)

            optimizer.step ()
            running_loss += loss.item () * inputs.size (0)

        # 验证阶段
        model.eval ()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad () :
            for inputs, labels in val_loader :
                if len (inputs) == 0 :
                    continue
                inputs, labels = inputs.to (Config.device), labels.to (Config.device)
                outputs = model (inputs)
                loss = criterion (outputs, labels)
                val_loss += loss.item () * inputs.size (0)

                _, predicted = torch.max (outputs, 1)
                total += labels.size (0)
                correct += (predicted == labels).sum ().item ()

        # 统计指标
        train_loss = running_loss / len (train_set)
        val_loss = val_loss / len (val_set)
        val_acc = 100 * correct / total

        # 早停判断
        if val_acc > best_acc :
            best_acc = val_acc
            no_improve = 0
            # 保存最佳模型
            torch.save (model.state_dict (), "body_type_resnet50.pth")
        else :
            no_improve += 1
            if no_improve >= patience :
                early_stop = True

        print (f"Epoch {epoch + 1}: "
               f"Train Loss: {train_loss:.4f} | "
               f"Val Loss: {val_loss:.4f} | "
               f"Val Acc {val_acc:.1f}%")

    print (f"训练完成，最佳验证准确率：{best_acc:.1f}%")


def collate_fn (batch) :
    valid_batch = [item for item in batch if item [1].item () != -1]
    return torch.utils.data.default_collate (valid_batch) if valid_batch else (torch.Tensor (), torch.LongTensor ())


# 预测函数
def predict (image_path, keypoints=None, model_path="body_type_resnet50.pth") :
    plt.rcParams ['font.sans-serif'] = ['SimHei']
    plt.rcParams ['axes.unicode_minus'] = False

    # 初始化模型
    model = CustomResNet ().to (Config.device)
    model.load_state_dict (torch.load (model_path, map_location=Config.device))
    model.eval ()

    # 读取和处理图像
    img = cv2.imread (image_path)
    if img is None :
        raise ValueError (f"无法读取图像：{image_path}")

    # 预处理
    processed = preprocess_image (cv2.cvtColor (img, cv2.COLOR_BGR2RGB))
    input_tensor = torch.FloatTensor (processed).unsqueeze (0).to (Config.device)

    # 预测
    with torch.no_grad () :
        outputs = model (input_tensor)
        probabilities = torch.nn.functional.softmax (outputs, dim=1)
        pred_class = torch.argmax (outputs).item ()

    # 可视化（保持不变）
    plt.figure (figsize=(14, 6))

    # 原始图像
    plt.subplot (1, 2, 1)
    plt.imshow (cv2.cvtColor (img, cv2.COLOR_BGR2RGB))
    plt.title ("原始图像")
    plt.axis ('off')

    # 带关键点的图像
    plt.subplot (1, 2, 2)
    if keypoints is not None :
        annotated_img = draw_keypoints (img, keypoints)
        plt.imshow (cv2.cvtColor (annotated_img, cv2.COLOR_BGR2RGB))
    else :
        plt.imshow (cv2.cvtColor (img, cv2.COLOR_BGR2RGB))
    plt.title (f"预测结果: {class_names [pred_class]}\n置信度: {probabilities [0] [pred_class]:.2f}")
    plt.axis ('off')

    plt.tight_layout ()
    plt.show ()

    return class_names [pred_class]


def generate_confusion_matrix_report (model_path="body_type_resnet50.pth",
                                      metadata_path="metadata.json",
                                      batch_size=16) :
    """生成并可视化混淆矩阵报告"""
    plt.rcParams ['font.sans-serif'] = ['SimHei']
    plt.rcParams ['axes.unicode_minus'] = False
    # 加载元数据
    with open (metadata_path) as f :
        metadata = json.load (f)

    # 重新划分验证集（保持与训练时相同的随机种子）
    _, val_meta = train_test_split (metadata, test_size=0.2, random_state=42)

    # 创建数据集和数据加载器
    val_set = CocoBodyDataset (val_meta)
    val_loader = DataLoader (
        val_set,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 初始化模型
    model = CustomResNet ().to (Config.device)
    model.load_state_dict (torch.load (model_path, map_location=Config.device))
    model.eval ()

    # 收集预测结果
    all_preds = []
    all_labels = []

    with torch.no_grad () :
        for inputs, labels in val_loader :
            if len (inputs) == 0 :
                continue
            inputs = inputs.to (Config.device)
            outputs = model (inputs)
            _, preds = torch.max (outputs, 1)

            all_preds.extend (preds.cpu ().numpy ())
            all_labels.extend (labels.cpu ().numpy ())

    # 生成混淆矩阵
    cm = confusion_matrix (all_labels, all_preds)
    report = classification_report (
        all_labels, all_preds,
        target_names=class_names.values (),
        output_dict=True
    )

    # 可视化
    plt.figure (figsize=(12, 10))
    sns.heatmap (cm, annot=True, fmt="d",
                 xticklabels=class_names.values (),
                 yticklabels=class_names.values (),
                 cmap="Blues")
    plt.title ("Confusion Matrix")
    plt.xlabel ("Predicted Label")
    plt.ylabel ("True Label")
    plt.xticks (rotation=45)
    plt.yticks (rotation=0)
    plt.tight_layout ()
    plt.show ()

    # 打印分类报告
    print ("\n详细分类指标：")
    print (classification_report (
        all_labels, all_preds,
        target_names=class_names.values ()
    ))

    return cm, report


# 顺序验证函数
def sequential_validation (max_samples=100) :
    # 初始化COCO API
    coco = COCO (Config.ann_file)

    # 获取所有包含人体的图像ID
    cat_ids = coco.getCatIds (catNms=['person'])
    img_ids = coco.getImgIds (catIds=cat_ids)

    validated_count = 0
    sample_index = 0  # 当前检查的索引

    while validated_count < max_samples and sample_index < len (img_ids) :
        current_img_id = img_ids [sample_index]
        sample_index += 1

        # 加载图像信息
        img_info_list = coco.loadImgs (current_img_id)
        if not img_info_list :
            continue

        img_info = img_info_list [0]
        img_path = os.path.join (Config.img_dir, img_info ['file_name'])

        # 跳过不存在文件
        if not os.path.exists (img_path) :
            continue

        # 获取标注信息
        ann_ids = coco.getAnnIds (imgIds=current_img_id, iscrowd=False)
        anns = coco.loadAnns (ann_ids)
        valid_anns = [ann for ann in anns if ann ["num_keypoints"] >= 10]

        if not valid_anns :
            continue

        try :
            # 获取关键点数据
            keypoints = np.array (valid_anns [0] ["keypoints"]).reshape (-1, 3)

            # 执行预测并显示关键点
            true_label = classify_body_type (keypoints)
            pred_label = predict (img_path, keypoints=keypoints)

            print (f"\n验证样本 {validated_count + 1}:")
            print (f"图像ID: {current_img_id}")
            print (f"文件路径: {img_path}")
            print (f"真实标签: {class_names [true_label]}")
            print (f"预测结果: {pred_label}")

            validated_count += 1

        except Exception as e :
            print (f"验证失败: {str (e)}")


# 关键点绘制函数
def draw_keypoints (image, keypoints, connections=None) :
    """
    在图像上绘制关键点和骨架连线
    :param image: BGR格式的numpy数组
    :param keypoints: COCO格式的关键点数组 (17, 3)
    :param connections: 需要连接的关节点对
    :return: 绘制后的图像
    """
    # COCO标准连接方式
    if connections is None :
        connections = [
            (0, 1), (0, 2),  # 头部
            (5, 6), (5, 7), (7, 9),  # 左臂
            (6, 8), (8, 10),  # 右臂
            (5, 11), (6, 12),  # 躯干
            (11, 13), (13, 15),  # 左腿
            (12, 14), (14, 16),  # 右腿
            (11, 12)  # 髋部
        ]

    img = image.copy ()

    # 绘制关键点
    for idx, (x, y, v) in enumerate (keypoints) :
        if v > 1 :  # COCO可见性判断：2表示可见
            color = (0, 0, 255) if idx in [5, 7, 9, 11, 13, 15] else (255, 0, 0)  # 左半身为红色，右半身为蓝色
            cv2.circle (img, (int (x), int (y)), 4, color, -1)

    # 绘制连线
    for (start, end) in connections :
        x1, y1, v1 = keypoints [start]
        x2, y2, v2 = keypoints [end]
        if v1 > 1 and v2 > 1 :
            cv2.line (img, (int (x1), int (y1)), (int (x2), int (y2)), (0, 255, 0), 2)

    return img


if __name__ == "__main__" :
    # train ()
    sequential_validation (max_samples=10)  # 验证时可以减少样本数量
    # generate_metadata ()
    # generate_confusion_matrix_report ()
