# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FunASR 中文语音转录项目。核心能力：**音频 → 带标点文本**，支持**说话人分离**和**声纹识别**。

## Environment

- Python 3.8.10 (系统 Python at `D:\python\python.exe`)
- 关键依赖：`funasr==1.0.27`, `modelscope==1.14.0`, `torch==2.4.1+cpu`
- 模型缓存：`C:\Users\Administrator\.cache\modelscope\hub\`
- 本机有代理问题，转录前必须设置：`export no_proxy="*"`（Windows 下 `$env:NO_PROXY="*"`）
- FFmpeg 已安装，路径在 `C:\Users\Administrator\Desktop\转码文件\ffmpeg.exe`

## Script Architecture

四层脚本体系，从简单到复杂：

| 层级 | 脚本 | 模型组合 | 输出 |
|------|------|------|------|
| L1 纯转录 | `transcribe.py` | paraformer + fsmn-vad + ct-punc | `.txt` |
| L2 说话人分离 | `transcribe_speaker.py` | L1 + CAM++ | `_speaker.txt` + `_speaker.json` |
| L3 对话形式 | `transcribe_dialogue.py` | L1 + CAM++ + 话轮合并 | `_dialogue.txt` + `_dialogue.json` |
| L4 批量处理 | `batch_transcribe.py` | 同 L3 | 批量 `_dialogue.txt` + `_batch_summary.txt` |

辅助脚本：
- `identify_speaker.py`：声纹注册/管理/跨会议匹配（独立于转录流程）
- `run_parallel.py`：N 路并行批量转录（拆分文件列表，调 `batch_transcribe.py`）

所有脚本共享同一个 `D:\yuyinxiangmu\` 工作目录，`cd /d/yuyinxiangmu` 后运行。

## Common Commands

### 单文件转录

```bash
# 纯文本（无说话人）
cd /d/yuyinxiangmu && export no_proxy="*" && python transcribe.py --audio <路径.mp3>
# 输出: 同目录下同名 .txt

# 对话形式 + 说话人分离（推荐用于多人会议）
cd /d/yuyinxiangmu && export no_proxy="*" && python transcribe_dialogue.py --audio <路径.mp3> --num-speakers 3
# 输出: 同目录下 _dialogue.txt，带时间轴和说话人标注

# 对话 + JSON + 声纹匹配
cd /d/yuyinxiangmu && export no_proxy="*" && python transcribe_dialogue.py --audio <路径.mp3> --num-speakers 3 --speaker-profile 蒸馏目标人 --output-json
```

### 批量转录

```bash
# 处理整个目录的所有 MP3（自动说话人分离）
cd /d/yuyinxiangmu && export no_proxy="*" && python batch_transcribe.py --dir <音频目录> --skip-existing

# 指定说话人数 + 声纹匹配 + 自定义输出目录
cd /d/yuyinxiangmu && export no_proxy="*" && python batch_transcribe.py --dir <音频目录> --num-speakers 3 --speaker-profile 蒸馏目标人 --txt-dir <输出目录> --skip-existing
```

### 声纹管理

```bash
# 注册说话人（需 30 秒以上纯人声参考音频，16kHz 单声道）
cd /d/yuyinxiangmu && export no_proxy="*" && python identify_speaker.py register --audio <参考音频.mp3> --name <名称>

# 追加参考声纹（覆盖不同录音环境）
cd /d/yuyinxiangmu && export no_proxy="*" && python identify_speaker.py add-reference --audio <参考音频.mp3> --name <名称>

# 查看已注册
cd /d/yuyinxiangmu && export no_proxy="*" && python identify_speaker.py list

# 删除
cd /d/yuyinxiangmu && export no_proxy="*" && python identify_speaker.py delete --name <名称>
```

## Output File Naming

- `transcribe.py` → `原文件名.txt`
- `transcribe_dialogue.py` → `原文件名_dialogue.txt` + `原文件名_dialogue.json`
- `batch_transcribe.py` → `原文件名_dialogue.txt` + `_batch_summary.txt`

## Voiceprint System

已注册声纹：「蒸馏目标人」（3 条参考声纹），存储在 `speaker_profiles/蒸馏目标人/`。

声纹匹配流程（`identify_speaker.py` 中的 `match_speakers_in_meeting`）：
1. 从 FunASR 返回的 `sentence_info` 中获取每个说话人的片段
2. 用 ffmpeg 裁剪音频片段 → CAM++ 提取声纹
3. 与已注册声纹做余弦相似度比对（top-3 段平均，默认阈值 0.65）
4. 灰度区间 [0.50, 0.65) 触发 Route A 更深采样重试
5. 匹配成功则替换标签（如 `说话人0` → `蒸馏目标人`）

关键约束：`match_speakers_in_meeting` 依赖 ffmpeg，运行前需确认 ffmpeg 在 PATH 中。

## Model Details

| 模型 | ModelScope ID | 功能 | 大小 |
|------|------|------|------|
| paraformer-zh | `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 语音识别 | ~1.3GB |
| fsmn-vad | `speech_fsmn_vad_zh-cn-16k-common-pytorch` | 语音活动检测 | ~30MB |
| ct-punc | `punc_ct-transformer_cn-en-common-vocab471067-large` | 标点恢复 | ~500MB |
| cam++ | 自动下载 | 说话人声纹识别 | ~30MB |

首次运行自动从 ModelScope 下载，需 `export no_proxy="*"` 避免代理问题。

## Critical Notes

- **不要并行跑 FunASR**：多个 FunASR 进程同时初始化会竞争模型文件，必须串行。批量用 `batch_transcribe.py`。
- `transcribe.py` 的断行逻辑：按中文标点断句（。！？；），每段 ≥40 字为一个段落，防止单行过长导致文件管理器预览卡死。
- GPU 自动检测：有 CUDA 则用 GPU，否则 CPU。本机当前为 CPU 模式。
- 转录时 `cd /d/yuyinxiangmu` 必须切换目录，因为 `transcribe.py` 中模型加载使用相对路径逻辑。
