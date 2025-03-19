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
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# 配置参数
class Config :
    data_path = r"C:\datasets\coco2017"
    ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 16
    num_classes = 3
    img_size = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")


# 类别映射
class_names = {
    0 : "倒三角形",
    1 : "矩形(H型)",
    2 : "椭圆形"
}


# 图像预处理
def preprocess_image (img) :
    img = cv2.resize (img, (Config.img_size [1], Config.img_size [0]))
    img = img.astype (np.float32) / 255.0
    img = (img - Config.mean) / Config.std
    return img.transpose (2, 0, 1)


# 数据集类（新增关键点处理）
class CocoBodyDataset (Dataset) :
    def __init__ (self, metadata) :
        self.metadata = metadata

    def __len__ (self) :
        return len (self.metadata)

    def __getitem__ (self, idx) :
        item = self.metadata [idx]
        img = cv2.imread (item ["img_path"])
        if img is None :
            return (torch.zeros (3, *Config.img_size),
                    torch.zeros (34),
                    torch.tensor (-1))

        # 图像处理
        img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
        img = preprocess_image (img)

        # 关键点归一化处理
        kpts = np.array (item ["keypoints"])
        img_w = item ["img_width"]
        img_h = item ["img_height"]
        kpt_coords = kpts [:, :2].astype (np.float32)
        kpt_coords [:, 0] /= img_w  # X坐标归一化
        kpt_coords [:, 1] /= img_h  # Y坐标归一化
        kpt_coords = kpt_coords.flatten ()

        return (torch.FloatTensor (img),
                torch.FloatTensor (kpt_coords),
                torch.tensor (item ["label"], dtype=torch.long))


# 双输入模型
class DualPathResNet (nn.Module) :
    def __init__ (self) :
        super ().__init__ ()
        # 图像处理路径
        resnet = models.resnet50 (weights=None)
        self.img_features = nn.Sequential (
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

        # 关键点处理路径
        self.kpt_processor = nn.Sequential (
            nn.Linear (34, 256),
            nn.BatchNorm1d (256),
            nn.ReLU (),
            nn.Dropout (0.5),
            nn.Linear (256, 128)
        )

        # 特征融合分类器
        self.classifier = nn.Sequential (
            nn.Linear (2048 + 128, 512),
            nn.BatchNorm1d (512),
            nn.ReLU (),
            nn.Dropout (0.3),
            nn.Linear (512, Config.num_classes)
        )

    def forward (self, img, kpts) :
        # 图像特征提取
        img_feat = self.img_features (img)
        img_feat = torch.flatten (img_feat, 1)

        # 关键点特征提取
        kpt_feat = self.kpt_processor (kpts)

        # 特征融合
        combined = torch.cat ([img_feat, kpt_feat], dim=1)
        return self.classifier (combined)


# 元数据生成（新增尺寸保存）
def generate_metadata () :
    if os.path.exists ("metadata.json") :
        with open ("metadata.json") as f :
            return json.load (f)

    coco = COCO (Config.ann_file)
    metadata = []
    class_counts = {cls : 0 for cls in class_names.keys ()}
    target_per_class = 1000

    # 关键点筛选参数
    HIP_X_DIFF_RATIO = 0.25
    SHOULDER_X_DIFF_RATIO = 0.25
    MIN_VISIBILITY = 0

    for img_id in coco.getImgIds (catIds=coco.getCatIds ('person')) :
        if all (count >= target_per_class for count in class_counts.values ()) :
            break

        img_info = coco.loadImgs (img_id) [0]
        img_path = os.path.join (Config.img_dir, img_info ["file_name"])
        img_width = img_info ["width"]
        img_height = img_info ["height"]

        if not os.path.exists (img_path) :
            continue

        for ann in coco.loadAnns (coco.getAnnIds (imgIds=img_id, iscrowd=False)) :
            if ann ["num_keypoints"] >= 10 :
                kpts = np.array (ann ["keypoints"]).reshape (-1, 3)
                required_joints = [5, 6, 11, 12]
                if not all (kpts [i, 2] >= MIN_VISIBILITY for i in required_joints) :
                    continue

                # 姿势筛选逻辑...
                # 提取关键点坐标和可见性
                ls = kpts [5]  # 左肩 (x,y,v)
                rs = kpts [6]  # 右肩
                lh = kpts [11]  # 左髋
                rh = kpts [12]  # 右髋

                # 质量控制检查

                # 新增：计算肩宽和臀宽
                shoulder_width = np.linalg.norm(ls[:2] - rs[:2])
                hip_width = np.linalg.norm(lh[:2] - rh[:2])
                eps = 1e-6  # 防止除以零

                # 新增：性别判断逻辑
                shoulder_hip_ratio = shoulder_width / (hip_width + eps)
                if shoulder_hip_ratio < 0.9:  # 排除女性特征样本
                    continue
                # 1. 排除异常姿势
                if abs (ls [1] - rs [1]) > 30 :  # 肩部垂直差
                    continue
                if np.linalg.norm (lh [:2] - rh [:2]) < 10 :  # 髋部宽度
                    continue

                # 新增正面照判断逻辑
                # 2. 髋部左右位置差（侧面时x坐标差会较大）
                hip_x_diff = abs (lh [0] - rh [0])
                if hip_x_diff > img_width * HIP_X_DIFF_RATIO :
                    continue

                # 3. 肩部左右位置差
                shoulder_x_diff = abs (ls [0] - rs [0])
                if shoulder_x_diff > img_width * SHOULDER_X_DIFF_RATIO :
                    continue

                # 4. 躯干朝向判断（肩膀中心与髋部中心的x坐标差应较小）
                shoulder_center_x = (ls [0] + rs [0]) / 2
                hip_center_x = (lh [0] + rh [0]) / 2
                if abs (shoulder_center_x - hip_center_x) > img_width * 0.1 :
                    continue

                # （保持原有筛选逻辑不变）

                label = classify_body_type (kpts)
                if class_counts [label] < target_per_class :
                    metadata.append ({
                        "img_path" : img_path,
                        "label" : label,
                        "keypoints" : kpts.tolist (),
                        "img_width" : img_width,  # 新增尺寸保存
                        "img_height" : img_height
                    })
                    class_counts [label] += 1

    # 平衡采样
    balanced_metadata = []
    for cls in class_names :
        balanced_metadata += [m for m in metadata if m ["label"] == cls] [:target_per_class]

    with open ("metadata.json", "w") as f :
        json.dump (balanced_metadata, f)

    return balanced_metadata


# 训练流程（新增双输入处理）
def train () :
    metadata = generate_metadata ()
    train_meta, val_meta = train_test_split (metadata, test_size=0.2, random_state=42)

    train_set = CocoBodyDataset (train_meta)
    val_set = CocoBodyDataset (val_meta)

    # 修改collate_fn处理三元组
    def collate_fn (batch) :
        valid_batch = [item for item in batch if item [2].item () != -1]
        if not valid_batch :
            return (torch.Tensor (), torch.Tensor (), torch.LongTensor ())
        imgs, kpts, labels = zip (*valid_batch)
        return (torch.stack (imgs),
                torch.stack (kpts),
                torch.stack (labels))

    train_loader = DataLoader (
        train_set, batch_size=Config.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader (
        val_set, batch_size=Config.batch_size,
        num_workers=0, collate_fn=collate_fn
    )

    # 初始化双路径模型
    model = DualPathResNet ().to (Config.device)

    # 加载预训练权重
    pretrained = models.resnet50 (weights=ResNet50_Weights.IMAGENET1K_V1)
    model.img_features.load_state_dict (pretrained.state_dict (), strict=False)

    # 优化器与损失函数
    optimizer = torch.optim.AdamW (model.parameters (), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss ()

    # 训练循环
    best_acc = 0.0
    for epoch in range (20) :
        model.train ()
        running_loss = 0.0

        # 训练步骤
        for imgs, kpts, labels in train_loader :
            imgs = imgs.to (Config.device)
            kpts = kpts.to (Config.device)
            labels = labels.to (Config.device)

            optimizer.zero_grad ()
            outputs = model (imgs, kpts)
            loss = criterion (outputs, labels)
            loss.backward ()
            optimizer.step ()

            running_loss += loss.item () * imgs.size (0)

        # 验证步骤
        model.eval ()
        val_loss = 0.0
        correct = 0
        with torch.no_grad () :
            for imgs, kpts, labels in val_loader :
                imgs = imgs.to (Config.device)
                kpts = kpts.to (Config.device)
                labels = labels.to (Config.device)

                outputs = model (imgs, kpts)
                loss = criterion (outputs, labels)
                val_loss += loss.item () * imgs.size (0)

                _, preds = torch.max (outputs, 1)
                correct += (preds == labels).sum ().item ()

        # 统计指标
        train_loss = running_loss / len (train_set)
        val_loss = val_loss / len (val_set)
        val_acc = 100 * correct / len (val_set)

        print (f"Epoch {epoch + 1}: "
               f"Train Loss: {train_loss:.4f} | "
               f"Val Loss: {val_loss:.4f} | "
               f"Val Acc {val_acc:.1f}%")

        # 保存最佳模型
        if val_acc > best_acc :
            best_acc = val_acc
            torch.save (model.state_dict (), "dual_path_model.pth")

    print (f"训练完成，最佳验证准确率：{best_acc:.1f}%")


# 预测函数（新增关键点输入）
def predict (image_path, keypoints, model_path="dual_path_model.pth") :
    # 初始化模型
    model = DualPathResNet ().to (Config.device)
    model.load_state_dict (torch.load (model_path))
    model.eval ()

    # 处理图像
    img = cv2.imread (image_path)
    img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
    processed_img = preprocess_image (img)
    img_tensor = torch.FloatTensor (processed_img).unsqueeze (0).to (Config.device)

    # 处理关键点
    img_h, img_w = img.shape [:2]
    kpts = np.array (keypoints) [:, :2].astype (np.float32)
    kpts [:, 0] /= img_w  # 归一化X坐标
    kpts [:, 1] /= img_h  # 归一化Y坐标
    kpt_tensor = torch.FloatTensor (kpts.flatten ()).unsqueeze (0).to (Config.device)

    # 预测
    with torch.no_grad () :
        outputs = model (img_tensor, kpt_tensor)
        probs = torch.nn.functional.softmax (outputs, dim=1)
        pred_class = torch.argmax (outputs).item ()

    # 可视化
    plt.figure (figsize=(12, 6))
    plt.subplot (1, 2, 1)
    plt.imshow (img)
    plt.title ("原始图像")

    plt.subplot (1, 2, 2)
    annotated_img = draw_keypoints (cv2.cvtColor (img, cv2.COLOR_RGB2BGR), keypoints)
    plt.imshow (cv2.cvtColor (annotated_img, cv2.COLOR_BGR2RGB))
    plt.title (f"预测: {class_names [pred_class]}\n置信度: {probs [0] [pred_class]:.2f}")

    plt.show ()
    return class_names [pred_class]


# 其余辅助函数保持不变...
# （包括classify_body_type、collate_fn、draw_keypoints等）
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
    train ()
    # 使用示例：
    # predict("test.jpg", keypoints=[[x1,y1,v1],...])