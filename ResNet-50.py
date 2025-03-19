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

# 类别标签映射
class_names = {
    0 : "沙漏型",
    1 : "倒三角型",
    2 : "梨型",
    3 : "标准型"
}


# 配置参数（修改输入尺寸）
class Config :
    data_path = r"C:\datasets\coco2017"
    ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 8  # 可适当增大batch_size
    num_classes = 4
    img_size = (224, 224)  # 修改为标准输入尺寸
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")


# 图像预处理函数（保持相同逻辑）
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
            nn.Dropout (0.5),
            nn.Linear (2048, 512),
            nn.ReLU (),
            nn.Dropout (0.3),
            nn.Linear (512, Config.num_classes)
        )

    def forward (self, x) :
        return self.base (x)

# 生成元数据（保持不变）
def generate_metadata () :
    if os.path.exists ("metadata.json") :
        with open ("metadata.json") as f :
            return json.load (f)

    coco = COCO (Config.ann_file)
    metadata = []
    class_counts = {cls : 0 for cls in class_names.keys ()}  # 初始化类别计数器
    target_per_class = 1000  # 每个类别目标样本数

    # 遍历所有人体图像
    for img_id in coco.getImgIds (catIds=coco.getCatIds ('person')) :
        if all (count >= target_per_class for count in class_counts.values ()) :
            break  # 所有类别都达到目标时提前终止

        img_info = coco.loadImgs (img_id) [0]
        img_path = os.path.join (Config.img_dir, img_info ["file_name"])

        if not os.path.exists (img_path) :
            continue

        # 遍历所有有效标注
        for ann in coco.loadAnns (coco.getAnnIds (imgIds=img_id, iscrowd=False)) :
            if ann ["num_keypoints"] >= 10 :
                kpts = np.array (ann ["keypoints"]).reshape (-1, 3)

                # 关键点有效性检查
                if all (kpts [i, 2] > 0 for i in [5, 6, 11, 12]) :
                    # 新增质量控制检查
                    left_shoulder = kpts[5]
                    right_shoulder = kpts[6]
                    left_hip = kpts[11]
                    right_hip = kpts[12]

                    # # 检查1: 左右关键点位置逻辑
                    # if (left_shoulder[0] < left_hip[0]) or (right_shoulder[0] > right_hip[0]):
                    #     continue  # 排除左肩在左髋左侧/右肩在右髋右侧的情况

                    # 检查2: 两肩高度差
                    if abs(left_shoulder[1] - right_shoulder[1]) > 30:
                        continue  # 排除肩膀倾斜严重的样本
                    # 检查3: 髋部宽度合理性（新增）
                    hip_width = np.linalg.norm(left_hip[:2] - right_hip[:2])
                    if hip_width < 10:  # 像素单位最小阈值
                        continue
                    label = classify_body_type (kpts)

                    # 检查是否已达到该类别上限
                    if class_counts [label] < target_per_class :
                        metadata.append ({
                            "img_path" : img_path,
                            "label" : label,
                            "keypoints" : kpts.tolist ()  # 保存关键点用于后续验证
                        })
                        class_counts [label] += 1

                        # 进度显示
                        print (f"\r采集进度: { {k : v for k, v in class_counts.items () if v > 0} }", end="")

                        # 检查是否所有类别都已满足
                        if all (count >= target_per_class for count in class_counts.values ()) :
                            break

    # 保存平衡后的元数据
    balanced_metadata = []
    for cls in class_names :
        balanced_metadata += [m for m in metadata if m ["label"] == cls] [:target_per_class]

    with open ("metadata.json", "w") as f :
        json.dump (balanced_metadata, f)

    print (
        f"\n最终样本分布: { {cls : len ([m for m in balanced_metadata if m ['label'] == cls]) for cls in class_names} }")
    return balanced_metadata


# 体型分类规则（保持不变）
def classify_body_type (kpts) :
    ls = kpts [5] [:2]
    rs = kpts [6] [:2]
    lh = kpts [11] [:2]
    rh = kpts [12] [:2]

    shoulder_width = np.linalg.norm (ls - rs)
    hip_width = np.linalg.norm (lh - rh)
    eps = 1e-6
    shr = shoulder_width / (hip_width + eps)
    whr = (0.7 * shoulder_width) / (hip_width + eps)
    # 计算新增特征：躯干长度比
    shoulder_center = (kpts[5][:2] + kpts[6][:2]) / 2
    hip_center = (kpts[11][:2] + kpts[12][:2]) / 2
    torso_ratio = np.linalg.norm(shoulder_center - hip_center) / (hip_width + eps)

    # 更新后的分类逻辑
    if shr > 1.15 and whr < 0.85 and torso_ratio > 1.2:
        return 0
    elif shr > 1.1 and whr >= 0.9:
        return 1
    elif shr < 0.95 and whr < 0.98:
        return 2
    else:  # 原标准型和苹果型合并
        return 3


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
    pretrained = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.base.load_state_dict(pretrained.state_dict(), strict=False)
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

    for epoch in range (20) :  # 增加最大epoch数
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


# 新增关键点绘制函数
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
    # train()
    sequential_validation(max_samples=100)  # 验证时可以减少样本数量
    # generate_metadata ()
