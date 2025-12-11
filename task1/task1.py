from ultralytics import YOLO
import os
import sys


DATASET_YAML = 'coco8.yaml'

def run_combined_task_simplified():
    # 1. 加载预训练模型
    # 'yolov8n.pt' 是 Nano 版本的模型，速度快，适合快速测试。
    print("--- 1. 加载 YOLOv8n 预训练模型 ---")
    model = YOLO('yolov8n.pt')

    # --- 任务 A: 验证模式 (获取指标和图表) ---
    print("\n--- 2. 启动验证模式 (Validation) ---")
    print("目标: 计算模型在 COCO8 数据集上的性能指标，如 mAP，并生成性能图表。")

    try:
        # model.val() 函数会执行以下操作：
        metrics = model.val(
            data=DATASET_YAML,
            imgsz=640,  # 图像尺寸
            plots=True,
            name="coco8_val_metrics"
        )

        print("\n--- 验证结果总结 (Task A) ---")
        # metrics.box 包含目标检测相关的指标
        print(f" mAP@0.5 (Mean Average Precision at IoU=0.5): {metrics.box.map50:.4f}")
        print(f" mAP@0.5:0.95 (在 IoU 0.5 到 0.95 范围的平均 mAP): {metrics.box.map:.4f}")

        # 构建并打印结果保存路径
        # YOLOv8 的结果通常保存在脚本所在目录下的 'runs/detect/指定的名称'
        val_save_dir = os.path.join(os.getcwd(), 'runs', 'detect', 'coco8_val_metrics')
        print(f" 性能指标图表和指标文件已保存到：{val_save_dir}")

    except Exception as e:
        print(f" 验证模式失败：{e}")
        # 如果验证失败，通常是因为数据下载或路径问题，继续尝试检测。

    # --- 任务 B: 检测模式 (保存所有带框图片) ---
    print("\n--- 3. 启动检测模式 (Prediction) ---")
    print("目标: 对 COCO8 验证集中的所有图片进行推理，并保存带有检测框的可视化结果。")

    # 注意：由于 model.val() 已经确保数据下载并知道其结构，我们可以直接指向验证集路径。
    # 对于 COCO8，其验证集图片路径通常位于 'datasets/coco8/images/val'。
    # 如果不确定，最简单的方式是直接使用验证集图片所在目录。
    # 这里我们使用一个通用的占位符，但在实际运行中，YOLO 倾向于在 val 模式下自动处理输入图片。
    #
    # **简化处理：直接使用一个 COCO8 验证集的图片路径作为检测的源头，
    #             或者对整个数据集目录进行检测（但更常用的是对单张图片或自定义目录）。**


    val_images_source = os.path.join('datasets', 'coco8', 'images', 'val')
    print(f"使用验证集图片目录作为检测源: {val_images_source}")

    try:
        # model.predict() 函数执行推理和可视化：
        results = model.predict(
            source=val_images_source,
            save=True,
            conf=0.25,
            name="coco8_val_predictions"
        )

        print("\n--- 检测结果总结 (Task B) ---")
        if results:
            num_images = len(results)
            # 构建并打印结果保存路径
            pred_save_dir = os.path.join(os.getcwd(), 'runs', 'detect', 'coco8_val_predictions')

            print(f" 成功对 {num_images} 张图片进行了目标检测和可视化。")
            print(f" 带有检测框的结果图片已保存到：{pred_save_dir}")
            print("请查看该目录下的所有图片文件，即可看到模型的检测效果。")

    except Exception as e:
        print(f"检测模式失败：{e}")

if __name__ == '__main__':
    # 运行整合任务
    run_combined_task_simplified()
    print("\n--- 任务完成 ---")