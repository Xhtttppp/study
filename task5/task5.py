import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import types

# ------------------------------------------------------
# 1.1. 自定义 FocalLoss 类（用于替换分类损失）

class FocalLoss(nn.Module):
    """
    Focal Loss 实现，用于替换默认的 BCE/Cross Entropy Loss。
    """

    def __init__(self, gamma=2., alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # 原始的 BCE Loss
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)

        # 计算 pt 和 Focal Loss
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()

# ------------------------------------------------------
# 2.1. 替换 Trainer 核心方法：只处理 Focal Loss 替换
# ------------------------------------------------------
def new_get_model_loss(self, preds, batch):
    """
    此方法只负责在损失计算后，用 Focal Loss 替换默认的分类损失。
    边界框损失 (Box Loss) 保持不变 (即 Ultralytics 默认的 IoU Loss)。
    """

    # 1. 调用原始损失函数 (计算 IoU/Box Loss 和原始 Cls Loss)
    # total_loss = IoU Loss + Cls Loss + DFL Loss
    # loss_items[1] = 原始 Cls Loss
    total_loss, loss_items = self.loss_fn(preds, batch, self.model)

    # 2. 计算 Focal Loss
    # preds[0] 是分类输出，batch[0] 是分类标签
    focal_cls_loss = self.focal_loss(preds[0], batch[0])

    # 3. 替换默认的分类损失项：
    # 从 total_loss 中减去原始 Cls Loss (loss_items[1])，然后加上 Focal Loss
    total_loss = total_loss - loss_items[1] + focal_cls_loss

    # 4. 替换 loss_items 中的分类损失，用于记录和显示
    loss_items[1] = focal_cls_loss.detach()

    return total_loss, loss_items



def patch_trainer_methods(trainer):
    """
    在 Trainer 实例化后 (on_train_start)，执行初始 Hook：
    1. 挂载 FocalLoss 实例。
    2. 替换 get_model_loss 方法。
    """
    print("\n---  Trainer 实例化完成。开始初始 Hook ---")

    # 1. 挂载 FocalLoss 实例 (用于 new_get_model_loss 使用)
    focal_loss = FocalLoss(gamma=2., alpha=0.25)
    trainer.focal_loss = focal_loss

    # 2. 替换 get_model_loss 方法
    trainer.get_model_loss = types.MethodType(new_get_model_loss, trainer)



def train_with_custom_loss(model, data_yaml, dataset_dir, epochs=100, imgsz=960, batch=16, device=0):
    # 注册回调 Hook
    model.add_callback('on_train_start', patch_trainer_methods)

    print(f"\n 开始训练：imgsz={imgsz}, batch={batch}, Cls Loss = FocalLoss")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        project=os.path.join(dataset_dir, "runs"),
        name="yolo11n_visdrone_focalloss_only",
        pretrained=False,
        amp=True,
        # 训练参数
        cutmix=0.0,
        mixup=0.0,
        mosaic=0.3,
        iou=0.45,
    )

    return results


# ------------------------------------------------------
# 4. main 函数
# ------------------------------------------------------
def main():
    dataset_dir = r"E:\python\pycharm\comsen_yoyo\task5\VisDrone2019-DET-train"
    data_yaml = os.path.join(dataset_dir, "data.yaml")

    if not os.path.exists(data_yaml):
        print(" ERROR: data.yaml 未找到！请检查路径：", data_yaml)
        return

    print("\n=== Loading YOLO11n model ===")
    model = YOLO("yolo11n.yaml")

    TARGET_IMG_SIZE = 960

    results = train_with_custom_loss(
        model=model,
        data_yaml=data_yaml,
        dataset_dir=dataset_dir,
        epochs=100,
        imgsz=TARGET_IMG_SIZE,
        batch=6,
        device=0,
    )

    print("\n=== Start Validation ===")
    model.val(data=data_yaml, imgsz=TARGET_IMG_SIZE, conf=0.01, iou=0.45)

    print("\n=== Training & Validation Done! ===")


if __name__ == "__main__":
    main()