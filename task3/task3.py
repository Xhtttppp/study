from ultralytics import YOLO
import os

# 设置训练和数据集的常量
DATA_YAML_PATH = 'E:/python/pycharm/comsen_yoyo/task3/PPE_Detection.v1i.yolov8/data.yaml'
ARCHITECTURE_YAML = 'yolov8n.yaml'
EPOCHS = 100
BATCH_SIZE = 16
DEVICE_ID = 0  # 假设使用第一个CUDA设备

if __name__ == '__main__':
    try:
        # --- 1. 加载模型架构 (从头开始训练) ---
        print(f"--- 1. 加载 YOLOv8n 网络架构: {ARCHITECTURE_YAML} ---")
        # 此时 model 对象仅包含网络结构，没有训练权重
        model = YOLO(ARCHITECTURE_YAML)

        # --- 2. 模型训练 ---
        print("\n--- 2. 启动模型训练 (从随机权重开始) ---")
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=640,
            batch=BATCH_SIZE,
            workers=4,
            name='custom_yolo_scratch',  # 更改名称以区分从头训练
            device=DEVICE_ID,
            #  关键修改: 确保从随机权重开始，而不是加载预训练权重
            pretrained=False
        )

        # --- 3. 验证模型在 'val' 验证集上的性能 ---
        print("\n--- 3. 验证模型在 'val' 验证集上的性能 ---")
        # model.val() 默认使用 data.yaml 中的 'val' 路径
        metrics_val = model.val()

        print("\n--- 验证集评估结果 ---")
        print(f" Val mAP@0.5 (Mean Average Precision at IoU=0.5): {metrics_val.box.map50:.4f}")
        print(f" Val mAP@0.5:0.95 (平均 mAP): {metrics_val.box.map:.4f}")

        # --- 4. 评估模型在 'test' 测试集上的性能 ---
        print("\n--- 4. 评估模型在 'test' 测试集上的性能 ---")
        # 明确指定 split='test'，使用 data.yaml 中定义的 'test' 路径
        metrics_test = model.val(split='test')

        print("\n--- 测试集评估结果 ---")
        print(f" Test mAP@0.5 (Mean Average Precision at IoU=0.5): {metrics_test.box.map50:.4f}")
        print(f" Test mAP@0.5:0.95 (平均 mAP): {metrics_test.box.map:.4f}")

        # 打印结果保存路径
        train_dir = os.path.join('runs', 'detect', 'custom_yolo_scratch')
        print(f"\n 训练和评估结果已保存到：{train_dir}")
        print(f"  最佳权重文件位于：{os.path.join(train_dir, 'weights', 'best.pt')}")


    except Exception as e:
        # 捕获并打印错误信息
        print(f"\n 任务发生错误: {e}")
        # 建议在实际项目中加入更详细的错误处理，例如检查文件路径是否存在