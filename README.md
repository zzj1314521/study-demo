# FunASR 语音转文本（ASR）项目

基于 [FunASR](https://github.com/modelscope/FunASR) 的中文语音转文本工具，使用 Paraformer 模型 + VAD 语音检测 + 标点恢复，支持将音频文件自动转录为带标点的纯文本。

## 环境要求

- **Python 3.9+**（推荐 3.10+）
- **FFmpeg**（用于音频格式转换）
- 网络连接（首次运行需下载模型）

## 快速开始

### 1. 克隆项目并进入目录

```bash
cd yuyinxiangmu
```

### 2. 创建并激活虚拟环境

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install funasr torch torchaudio
```

> 如果遇到 SSL/网络错误，可尝试设置代理或使用镜像源：
> ```bash
> pip install funasr torch torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 4. 确认 FFmpeg 已安装

```bash
ffmpeg -version
```

如果没有安装：
- **Windows**: 从 [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 下载，解压后将 `bin` 目录加入系统 PATH
- **macOS**: `brew install ffmpeg`
- **Linux (Ubuntu/Debian)**: `sudo apt install ffmpeg`

### 5. 运行转录

```bash
python transcribe.py --audio <你的音频文件路径>
```

**示例:**
```bash
python transcribe.py --audio test_audio.mp3
```

输出：在与输入音频相同的目录下生成同名的 `.txt` 文件，包含识别出的文本。

## 模型说明

本脚本使用以下 FunASR 模型组合：

| 模型 | 标识 | 用途 |
|------|------|------|
| Paraformer | `paraformer-zh` | 中文语音识别（核心模型） |
| FSMN-VAD | `fsmn-vad` | 语音活动检测（自动切分语音段） |
| CT-Transformer | `ct-punc` | 标点恢复（自动添加标点符号） |

首次运行时，脚本会自动从 ModelScope 下载模型文件（总计约 **500 MB**），请保持网络畅通并耐心等待下载完成。

## 支持的音频格式

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)
- 其他 FFmpeg 支持的常见音频格式

## 使用技巧

- **长音频**: VAD 模型会自动切分语音片段，适合处理会议录音、讲座等长音频
- **GPU 加速**: 如果系统有 NVIDIA GPU 并安装了 CUDA 版本的 PyTorch，脚本会自动启用 GPU 加速
- **纯 CPU**: 无 GPU 环境下依然可以运行，转录速度在可接受范围内

## 故障排除

### 模型下载失败

如果从 ModelScope 下载模型超时或失败，可以尝试：

1. 设置环境变量以绕过某些网络代理：
   ```powershell
   # Windows PowerShell
   $env:NO_PROXY="*"
   python transcribe.py --audio audio.mp3
   ```

2. 手动下载模型文件放置在 ModelScope 缓存目录下。

### 识别结果为空

- 确认音频文件包含人声（不是纯音乐或静音）
- 检查音频采样率和编码格式是否正常
- 尝试将音频转为 16kHz WAV 格式：`ffmpeg -i input.mp3 -ar 16000 output.wav`

### Python 版本兼容性

本项目需要 **Python 3.9 或更高版本**。如果遇到 `'type' object is not subscriptable` 等错误，请升级 Python 版本。

## 脚本说明

| 脚本 | 功能 | 输出 |
|------|------|------|
| `transcribe.py` | 基础转录（纯文本，无说话人） | `xxx.txt` |
| `transcribe_speaker.py` | 转录 + 说话人分离 | `xxx_speaker.txt` + `.json` |
| `transcribe_dialogue.py` | 转录 + 对话形式（推荐） | `xxx_dialogue.txt` + `.json` |
| `batch_transcribe.py` | 批量处理目录下所有 MP3 | 多个 `_dialogue.txt` + 汇总 |
| `identify_speaker.py` | 声纹注册 + 跨会议说话人标记 | `speaker_profiles/` |

### 单文件转录

```bash
# 基础版（纯文本）
python transcribe.py --audio meeting.mp3

# 对话版（说话人分离 + 时间轴，推荐）
python transcribe_dialogue.py --audio meeting.mp3 --num-speakers 3

# 对话版 + JSON 输出
python transcribe_dialogue.py --audio meeting.mp3 --num-speakers 3 --output-json
```

### 🆕 跨会议说话人标记

在批量会议中统一标记同一个人（如老板、关键客户），只需提供一段参考音频：

```bash
# 第 1 步：注册参考说话人（只需做一次）
python identify_speaker.py register --audio 参考音频.mp3 --name 蒸馏目标人

# 第 2 步：查看已注册
python identify_speaker.py list

# 第 3 步：转录时自动标记
python transcribe_dialogue.py --audio meeting.mp3 --num-speakers 3 --speaker-profile 蒸馏目标人

# 批量模式自动标记
python batch_transcribe.py --dir D:\音频目录 --speaker-profile 蒸馏目标人 --skip-existing
```

> **参考音频要求**：16kHz 单声道，≥30 秒纯人声。"EV录屏"等软件录制的音频需先用 ffmpeg 转格式：
> ```bash
> ffmpeg -i 录屏.mp3 -ar 16000 -ac 1 参考_16k.mp3
> ```

### 批量转录

```bash
# 处理整个目录的所有 MP3（自动识别说话人数）
python batch_transcribe.py --dir D:\音频目录

# 指定 3 位说话人 + 跳过已处理的
python batch_transcribe.py --dir D:\音频目录 --num-speakers 3 --skip-existing

# 全套参数
python batch_transcribe.py --dir D:\音频目录 --num-speakers 3 --merge-gap 1.5 --output-json --skip-existing
```

## 项目结构

```
yuyinxiangmu/
├── venv/                    # Python 虚拟环境
├── transcribe.py            # 基础转录脚本
├── transcribe_speaker.py    # 说话人分离脚本
├── transcribe_dialogue.py   # 对话形式转录（推荐）
├── batch_transcribe.py      # 批量处理脚本
├── identify_speaker.py      # 声纹注册 + 跨会议识别
├── speaker_profiles/        # 已注册的说话人声纹
│   └── 蒸馏目标人/          #   embedding.npy + info.json
├── test_audio.mp3           # 测试用中文音频
└── README.md                # 本说明文档
```

## 许可证

本项目采用 MIT 许可证。FunASR 模型的使用请参考 [ModelScope License](https://www.modelscope.cn)。
