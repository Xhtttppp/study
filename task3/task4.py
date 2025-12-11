from ultralytics import YOLO

def train_yolo12():
    # 数据集配置
    DATA_YAML = r"E:/python/pycharm/comsen_yoyo/task4/PPE_Detection.v1i.yolov8/data.yaml"

    # 1. 加载 YOLO12 模型
    model = YOLO("yolo12n.yaml")  # 或 yolov12s.pt / yolov12m.pt

    # 2. 训练模型
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        name="yolo12_ppe",
        pretrained = False
    )

    print(" 训练已完成！YOLO 自动生成可视化图像在：runs/detect/yolo12_ppe/")

    # 3. 验证集评估
    metrics_val = model.val(split="val")
    print("\n--- 验证集评估 ---")
    print(f"mAP50: {metrics_val.box.map50:.4f}")
    print(f"mAP50-95: {metrics_val.box.map:.4f}")

    # 4. 测试集评估
    metrics_test = model.val(split="test")
    print("\n--- 测试集评估 ---")
    print(f"mAP50: {metrics_test.box.map50:.4f}")
    print(f"mAP50-95: {metrics_test.box.map:.4f}")

if __name__ == "__main__":
    train_yolo12()
