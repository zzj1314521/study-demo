#!/usr/bin/env python3
"""
FunASR 语音转文本（ASR）转录脚本

使用 Paraformer 中文模型 + VAD（语音活动检测）+ 标点恢复，
将音频文件中的语音转换为带标点的纯文本。

用法:
    python transcribe.py --audio <音频文件路径>

模型说明:
    - paraformer-zh : 中文 Paraformer 语音识别模型（核心）
    - fsmn-vad      : 语音活动检测（自动切分长音频中的语音段）
    - ct-punc       : 标点恢复模型（为识别结果加上标点符号）

首次运行时，FunASR 会自动从 ModelScope 下载模型（约几百 MB），
请保持网络畅通并耐心等待。
"""

import argparse
import os
import sys
import time

# --- Windows 控制台 UTF-8 支持 ---
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def transcribe_audio(audio_path: str) -> str:
    """
    对音频文件进行语音识别，返回带标点的纯文本。

    Args:
        audio_path: 音频文件的路径（支持 wav, mp3, flac, m4a 等常见格式）

    Returns:
        识别出的纯文本字符串

    Raises:
        FileNotFoundError: 音频文件不存在
        Exception: 模型下载或推理过程中的其他错误
    """
    # --- 检查音频文件是否存在 ---
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    if not os.path.isfile(audio_path):
        raise ValueError(f"路径不是有效的文件: {audio_path}")

    # --- 延迟导入，避免 argparse --help 时触发模型加载 ---
    from funasr import AutoModel

    print("=" * 60)
    print("正在初始化 FunASR 模型（首次运行将自动下载模型）...")
    print("模型: paraformer-zh (语音识别) + fsmn-vad (语音检测) + ct-punc (标点)")
    print("=" * 60)

    try:
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            # 自动检测设备：有 GPU 则用 GPU，否则用 CPU
            device="cuda:0" if _has_cuda() else "cpu",
            disable_update=True,  # 禁用版本更新检查
        )
    except Exception as e:
        print(f"\n[错误] 模型初始化失败: {e}", file=sys.stderr)
        print("提示: 请检查网络连接，模型需要从 ModelScope 下载。", file=sys.stderr)
        sys.exit(1)

    print("\n模型就绪，开始转录...")
    print(f"音频文件: {audio_path}")
    print("-" * 60)

    start_time = time.time()

    try:
        result = model.generate(input=audio_path)
    except Exception as e:
        print(f"\n[错误] 转录过程出错: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start_time

    # --- 解析结果 ---
    # FunASR 返回列表，每段包含 "text" 字段
    if not result:
        print("[错误] 模型返回了空结果。", file=sys.stderr)
        sys.exit(1)

    # 拼接所有识别片段
    texts = []
    for segment in result:
        text = segment.get("text", "").strip()
        if text:
            texts.append(text)

    raw_text = "".join(texts)

    # 按句子断行，避免单行过长导致文件管理器预览卡死
    import re
    full_text = re.sub(r"(。|！|？|；)", r"\1\n", raw_text)
    # 合并过短的行（VAD 切分可能导致碎片化）
    lines = full_text.split("\n")
    merged = []
    buf = ""
    for line in lines:
        if not line.strip():
            continue
        buf += line
        if len(buf) >= 40:  # 满 40 字作为一个段落
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    full_text = "\n".join(merged) + "\n"

    if not full_text:
        print("[警告] 识别结果为空，请检查音频是否包含有效语音。", file=sys.stderr)
        full_text = "（未识别到语音内容）"

    print(f"\n转录完成，耗时 {elapsed:.1f} 秒。")

    return full_text


def _has_cuda() -> bool:
    """检查是否有可用的 CUDA GPU。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="FunASR 语音转文本（ASR）转录工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python transcribe.py --audio test_audio.wav
    python transcribe.py --audio ~/Downloads/meeting.mp3
        """,
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="要转录的音频文件路径（支持 wav, mp3, flac, m4a 等格式）",
    )

    args = parser.parse_args()
    audio_path = args.audio

    # 执行转录
    text = transcribe_audio(audio_path)

    # --- 输出结果 ---
    # 确定输出文件路径：与输入音频同目录、同名，扩展名为 .txt
    base_name = os.path.splitext(audio_path)[0]
    output_path = base_name + ".txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"结果已保存至: {output_path}")
    print("=" * 60)
    print("识别结果:")
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
