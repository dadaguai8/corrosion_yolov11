# corrosion_yolov11

基于 YOLO11 分割模型的工业表面腐蚀检测系统。项目提供 Flask 后端和原生前端页面，可用于图片、图片文件夹、视频和摄像头画面的腐蚀识别，并支持模型切换、结果保存、检测统计和报告查看。

## 功能特性

- 支持 YOLO11 `.pt`、`.onnx`、`.engine` 模型文件
- 支持单张图片检测
- 支持图片文件夹批量检测
- 支持视频文件检测
- 支持摄像头实时帧检测
- 支持置信度和 IOU 阈值调节
- 输出检测目标数量、平均置信度、腐蚀类型分布、面积占比和严重程度
- 支持检测结果下载
- 支持 HTML 检测报告和 Word 报告导出

## 项目结构

```text
.
├── backend/
│   └── app.py              # Flask 后端服务与 YOLO11 推理逻辑
├── frontend/
│   ├── index.html          # 前端页面
│   ├── app.js              # 页面交互与 API 请求
│   └── styles.css          # 页面样式
├── models/
│   ├── YOLO11n-seg.pt      # 默认候选模型
│   └── YOLO11s-seg.pt      # 备用模型
└── README.md
```

运行后，后端会自动创建运行所需的临时目录，例如 `backend/uploads/`、`backend/outputs/` 和 `backend/.ultralytics/`。

## 使用环境

本项目使用 YOLO11 分割模型作为核心检测模型，当前仓库内置模型为：

- `models/YOLO11n-seg.pt`
- `models/YOLO11s-seg.pt`

当前使用的运行环境为 Conda 环境 `yolov11`：

| 环境 / 库 | 当前版本 | 说明 |
| --- | --- | --- |
| Conda 环境 | yolov11 | 项目运行环境 |
| Python | 3.10.19 | `D:\Anaconda\envs\yolov11\python.exe` |
| Flask | 3.1.3 | 后端 Web 服务 |
| ultralytics | 8.3.40 | YOLO11 推理框架 |
| opencv-python | 4.8.1.78 | 图像、视频读取与处理 |
| numpy | 1.26.4 | 数值计算与图像数组处理 |
| torch | 2.0.0+cu118 | PyTorch 推理后端，CUDA 11.8 版本 |
| torchvision | 0.15.1+cu118 | PyTorch 视觉工具包，CUDA 11.8 版本 |
| torchaudio | 未安装 | 非本项目核心依赖 |

> 说明：`backend/app.py` 中直接依赖 `flask`、`opencv-python`、`numpy`、`ultralytics`。其中 `ultralytics` 会使用 PyTorch 作为推理后端，因此运行 YOLO11 检测前需要安装匹配的 PyTorch 环境。

## 环境安装

建议使用 Conda 环境 `yolov11` 运行：

```powershell
conda activate yolov11
python backend\app.py
```

如需重新创建环境，可参考以下依赖版本安装：

```powershell
conda create -n yolov11 python=3.10
conda activate yolov11
pip install flask==3.1.3 ultralytics==8.3.40 opencv-python==4.8.1.78 numpy==1.26.4
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

如果使用 CPU 环境，也可以安装 CPU 版本 PyTorch，但视频检测和大模型推理速度会较慢。

## 启动项目

在项目根目录执行：

```powershell
conda activate yolov11
python backend\app.py
```

服务启动后访问：

```text
http://127.0.0.1:5000
```

## 使用说明

1. 打开页面后，系统会自动加载 `models/YOLO11n-seg.pt`，如果不存在则使用候选路径中的其他模型。
2. 可在左侧模型区域切换已有模型，或上传新的 `.pt`、`.onnx`、`.engine` 权重文件。
3. 选择检测源：图片、文件夹、视频或摄像头。
4. 根据需要调整置信度和 IOU 阈值。
5. 点击开始检测，查看画布标注结果和右侧统计信息。
6. 检测完成后可下载结果或查看检测报告。

## API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/model` | 获取当前模型、可用模型和模型指标 |
| `POST` | `/api/model` | 上传或切换模型 |
| `POST` | `/api/detect/image` | 单张图片检测 |
| `POST` | `/api/detect/batch` | 批量图片检测 |
| `POST` | `/api/detect/batch/finalize` | 汇总批量检测结果 |
| `POST` | `/api/detect/video` | 视频检测 |
| `POST` | `/api/detect/frame` | 摄像头或前端帧检测 |
| `GET` | `/api/download/result/<result_id>` | 下载检测结果 |
| `GET` | `/api/report/<result_id>` | 查看检测报告 |
| `GET` | `/api/report/<result_id>/word` | 导出 Word 报告 |

## 模型文件

当前仓库包含以下模型文件：

- `models/YOLO11n-seg.pt`
- `models/YOLO11s-seg.pt`

如需替换模型，可将训练好的 YOLO11 分割权重放入 `models/` 目录，或通过页面上传模型文件。

## 注意事项

- 首次加载模型和首次推理可能较慢，属于正常现象。
- 视频检测会消耗较多 CPU/GPU 资源。
- 检测输出文件会保存在 `backend/outputs/`，上传文件会保存在 `backend/uploads/`。
- 如界面中文出现乱码，请确认文件编码和浏览器解析均为 UTF-8。

## GitHub

仓库地址：

```text
https://github.com/dadaguai8/corrosion_yolov11
```
