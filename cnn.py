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

# 类别标签映射
class_names = {
    0 : "沙漏型",
    1 : "倒三角型",
    2 : "梨型",
    3 : "苹果型",
    4 : "标准型"
}

# 配置参数
class Config :
    data_path = r"C:\datasets\coco2017"
    ann_file = os.path.join (data_path, "annotations/person_keypoints_train2017.json")
    img_dir = os.path.join (data_path, "train2017")
    batch_size = 4
    num_classes = 5
    img_size = (256, 192)  # (height, width)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")


# 图像预处理函数
def preprocess_image (img) :
    # 调整尺寸
    img = cv2.resize (img, (Config.img_size [1], Config.img_size [0]))  # (width, height)
    # 标准化
    img = img.astype (np.float32) / 255.0
    img = (img - Config.mean) / Config.std
    return img.transpose (2, 0, 1)  # HWC to CHW


# 数据集类
class CocoBodyDataset (Dataset) :
    def __init__ (self, metadata) :
        self.metadata = metadata

    def __len__ (self) :
        return len (self.metadata)

    def __getitem__ (self, idx) :
        item = self.metadata [idx]

        # 读取图像
        img = cv2.imread (item ["img_path"])
        if img is None :
            print (f"无法读取图像：{item ['img_path']}")
            return torch.zeros (3, *Config.img_size), torch.tensor (-1)

        img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
        img = preprocess_image (img)

        return torch.FloatTensor (img), torch.tensor (item ["label"], dtype=torch.long)


# 改进的CNN模型结构
class EnhancedBodyCNN (nn.Module) :
    def __init__ (self) :
        super ().__init__ ()
        self.feature_extractor = nn.Sequential (
            # 输入: 3x256x192
            nn.Conv2d (3, 64, 3, padding=1),  # 64x256x192
            nn.BatchNorm2d (64),
            nn.ReLU (),
            nn.MaxPool2d (2),  # 64x128x96

            nn.Conv2d (64, 128, 3, padding=1),  # 128x128x96
            nn.BatchNorm2d (128),
            nn.ReLU (),
            nn.MaxPool2d (2),  # 128x64x48

            nn.Conv2d (128, 256, 3, padding=1),  # 256x64x48
            nn.BatchNorm2d (256),
            nn.ReLU (),
            nn.MaxPool2d (2),  # 256x32x24

            nn.Conv2d (256, 512, 3, padding=1),  # 512x32x24
            nn.BatchNorm2d (512),
            nn.ReLU (),
            nn.AdaptiveAvgPool2d ((6, 4))  # 512x6x4
        )

        self.classifier = nn.Sequential (
            nn.Dropout (0.5),
            nn.Linear (512 * 6 * 4, 1024),
            nn.ReLU (),
            nn.BatchNorm1d (1024),
            nn.Dropout (0.3),
            nn.Linear (1024, 512),
            nn.ReLU (),
            nn.BatchNorm1d (512),
            nn.Linear (512, Config.num_classes)
        )

    def forward (self, x) :
        x = self.feature_extractor (x)
        x = x.view (x.size (0), -1)
        return self.classifier (x)


# 生成元数据（修正采样逻辑）
def generate_metadata () :
    if os.path.exists ("metadata.json") :
        with open ("metadata.json") as f :
            return json.load (f)

    coco = COCO (Config.ann_file)
    metadata = []

    for img_id in coco.getImgIds (catIds=coco.getCatIds ('person')) :
        img_info = coco.loadImgs (img_id) [0]
        img_path = os.path.join (Config.img_dir, img_info ["file_name"])

        if not os.path.exists (img_path) :
            continue

        # 遍历所有有效的人体标注
        for ann in coco.loadAnns (coco.getAnnIds (imgIds=img_id, iscrowd=False)) :
            if ann ["num_keypoints"] >= 10 :
                kpts = np.array (ann ["keypoints"]).reshape (-1, 3)
                if all (kpts [i, 2] > 0 for i in [5, 6, 11, 12]) :
                    metadata.append ({
                        "img_path" : img_path,
                        "label" : classify_body_type (kpts)
                    })
                    # 保留break表示每图只取第一个有效实例
                    # 删除break可采集所有有效实例
                    break

                    # 保存完整元数据
    with open ("metadata.json", "w") as f :
        json.dump (metadata [:5000], f)  # 示例取2000个样本
    return metadata [:5000]


# 体型分类规则（保持原样）
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

    if shr > 1.15 and whr < 0.85 :
        return 0
    elif shr > 1.1 and whr >= 0.9 :
        return 1
    elif shr < 0.95 and whr < 0.98 :
        return 2
    elif 0.98 <= shr <= 1.05 and whr > 0.95 :
        return 3
    else :
        return 4


# 训练流程（仅修改模型初始化）
def train () :
    metadata = generate_metadata ()
    train_meta, val_meta = train_test_split (metadata, test_size=0.2, random_state=42)

    train_set = CocoBodyDataset (train_meta)
    val_set = CocoBodyDataset (val_meta)

    def collate_fn (batch) :
        valid_batch = [item for item in batch if item [1].item () != -1]
        if not valid_batch :
            return torch.Tensor (), torch.LongTensor ()
        return torch.utils.data.default_collate (valid_batch)

    train_loader = DataLoader (
        train_set, batch_size=Config.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader (
        val_set, batch_size=Config.batch_size,
        num_workers=0, collate_fn=collate_fn
    )

    # 使用新模型
    model = EnhancedBodyCNN ().to (Config.device)
    criterion = nn.CrossEntropyLoss ()
    optimizer = torch.optim.Adam (model.parameters (), lr=1e-4, weight_decay=1e-4)

    for epoch in range (5) :  # epoch=5
        model.train ()
        running_loss = 0.0  # 添加loss统计
        for inputs, labels in train_loader :
            if len (inputs) == 0 :
                continue
            inputs = inputs.to (Config.device)
            labels = labels.to (Config.device)

            optimizer.zero_grad ()
            outputs = model (inputs)
            loss = criterion (outputs, labels)
            loss.backward ()
            optimizer.step ()
            # 更新running loss
            running_loss += loss.item () * inputs.size (0)
            # 计算epoch平均loss
            epoch_loss = running_loss / len (train_loader.dataset)

        model.eval ()
        total, correct = 0, 0
        with torch.no_grad () :
            for inputs, labels in val_loader :
                if len (inputs) == 0 :
                    continue
                inputs = inputs.to (Config.device)
                labels = labels.to (Config.device)

                outputs = model (inputs)
                _, predicted = torch.max (outputs, 1)
                total += labels.size (0)
                correct += (predicted == labels).sum ().item ()

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Acc {100 * correct / total:.1f}%")
    # 训练完成后保存模型
    torch.save(model.state_dict(), "body_type_cnn.pth")
    print("模型已保存为 body_type_cnn.pth")


# 修改后的预测函数（添加关键点绘制）
def predict (image_path, keypoints=None, model_path="body_type_cnn.pth") :
    # 设置中文字体
    plt.rcParams ['font.sans-serif'] = ['SimHei']
    plt.rcParams ['axes.unicode_minus'] = False


    # 加载模型
    model = EnhancedBodyCNN ().to (Config.device)
    model.load_state_dict (torch.load (model_path, map_location=Config.device))
    model.eval ()

    # 读取图像
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

    # 可视化
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


# 修改后的顺序验证函数
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
    # train ()
    # 验证训练好的模型
    # validate_model()
    sequential_validation (max_samples=100)