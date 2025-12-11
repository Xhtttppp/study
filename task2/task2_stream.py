import cv2
import requests
from ultralytics import YOLO

# 下载视频文件
def download_video(url, output_path='downloaded_video.mp4'):
    """
    使用requests下载视频文件
    """
    print("开始下载视频...")
    response = requests.get(url, stream=True)  # 通过requests库获取视频流
    with open(output_path, 'wb') as file:  # 打开本地文件，准备写入
        for chunk in response.iter_content(chunk_size=1024):  # 按块下载
            if chunk:
                file.write(chunk)  # 将下载的视频内容写入到本地文件
    print(f"视频下载完成，保存在：{output_path}")
    return output_path  # 返回下载的视频文件路径


# 使用YOLO模型进行目标检测
def run_video_detection(source, model_path, confidence_threshold=0.5, save_video=False):
    """
    使用 YOLO 模型对视频文件进行目标检测。
    """
    model = YOLO(model_path)  # 加载YOLO模型，传入模型文件路径
    # 参数解释：
    # - `model_path`：传入预训练模型的路径（例如`yolov8n.pt`），该参数告诉YOLO加载哪个模型进行目标检测。

    cap = cv2.VideoCapture(source)  # 打开视频文件或摄像头流
    if not cap.isOpened():  # 如果无法打开视频源，打印错误信息并结束函数
        print(f" 错误: 无法打开视频源: {source}")
        return

    # 获取视频的宽度、高度和帧率
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # 设置视频写入器（可选，若需要保存视频）
    if save_video:  # 如果需要保存视频结果
        video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 实时帧循环处理
    while cap.isOpened():
        success, frame = cap.read()  # 读取视频的每一帧
        if not success:  # 如果读取失败，说明视频结束或出现错误
            break  # 跳出循环，结束检测

        # 执行目标检测
        results = model.predict(source=frame, conf=confidence_threshold, imgsz=640, iou=0.5, verbose=False)
        # - `source=frame`：传入当前视频帧进行目标检测。
        # - `conf=confidence_threshold`：设置检测框的最小置信度，低于该置信度的框会被忽略。默认值为0.5。
        # - `imgsz=640`：设置输入图像的大小，YOLOv8模型通常需要输入640x640大小的图像。
        # - `iou=0.7`：设置IOU（交并比）阈值，通常用来过滤掉重叠太多的框。
        # - `verbose=False`：是否打印详细信息。

        # 获取带有检测框和标签的图像
        annotated_frame = results[0].plot()  # 获取带注释的图像（即检测框已绘制）

        # 保存视频（如果设置了保存）
        if save_video:
            video_writer.write(annotated_frame)

        # 退出机制：按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()  # 释放视频读取资源
    if save_video:  # 如果保存视频，释放视频写入资源
        video_writer.release()
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    print(" 目标检测任务完成，资源已释放。")


# 主程序执行部分
if __name__ == "__main__":
    # 1. 下载公开的视频（选择一个公开的视频URL）
    video_url = "https://www.w3schools.com/html/movie.mp4"  # 新的视频URL（真实场景视频）
    video_file = download_video(video_url, 'sample_video.mp4')  # 下载视频

    # 2. 使用YOLO模型进行目标检测
    if video_file:  # 如果视频下载成功
        model_path = 'yolov8n.pt'
        run_video_detection(
            source=video_file,  # 视频文件路径
            model_path=model_path,  # YOLO模型路径
            confidence_threshold=0.3,  # 目标检测置信度阈值
            save_video=True  # 保存检测结果
        )
