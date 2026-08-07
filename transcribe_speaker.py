#!/usr/bin/env python3
"""
FunASR 语音转文本 + 说话人分离脚本

在原有 ASR 管道基础上集成 CAM++ 说话人识别模型，
自动区分不同说话人并标注到转录结果中。

用法:
    python transcribe_speaker.py --audio <音频文件路径>
    python transcribe_speaker.py --audio <音频文件路径> --num-speakers 3
    python transcribe_speaker.py --audio <音频文件路径> --output-json

输出:
    - <音频名>_speaker.txt  : 带说话人标注的纯文本
    - <音频名>_speaker.json : 带时间戳和说话人的结构化 JSON（需 --output-json）

模型:
    - paraformer-zh : 中文语音识别
    - fsmn-vad      : 语音活动检测
    - ct-punc       : 标点恢复
    - cam++         : 说话人声纹识别（新增）

首次运行时约额外下载 ~30 MB 的 CAM++ 模型。
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# --- Windows 控制台 UTF-8 支持 ---
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS 或 MM:SS 格式。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def transcribe_with_speakers(
    audio_path: str,
    preset_spk_num: int | None = None,
    device: str = "cpu",
):
    """
    对音频文件进行语音识别 + 说话人分离。

    Args:
        audio_path: 音频文件路径
        preset_spk_num: 预设说话人数量（None 则自动估计）
        device: 计算设备

    Returns:
        list[dict]: 每个元素包含 {spk, start, end, text, sentence_info}
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    if not os.path.isfile(audio_path):
        raise ValueError(f"路径不是有效的文件: {audio_path}")

    from funasr import AutoModel

    print("=" * 70)
    print("正在初始化 FunASR 模型（含说话人分离）...")
    print("ASR: paraformer-zh | VAD: fsmn-vad | PUNC: ct-punc | SPK: cam++")
    print("=" * 70)

    # 构建额外参数
    extra_kwargs = {}
    if preset_spk_num is not None:
        extra_kwargs["preset_spk_num"] = preset_spk_num
        print(f"预设说话人数: {preset_spk_num}")

    try:
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            device=device,
            disable_update=True,
        )
    except Exception as e:
        print(f"\n[错误] 模型初始化失败: {e}", file=sys.stderr)
        print("提示: 请检查网络连接。", file=sys.stderr)
        sys.exit(1)

    print("\n模型就绪，开始转录 + 说话人分离...")
    print(f"音频文件: {audio_path}")
    print("-" * 70)

    start_time = time.time()

    try:
        result = model.generate(
            input=audio_path,
            return_spk_res=True,
            **extra_kwargs,
        )
    except Exception as e:
        print(f"\n[错误] 转录过程出错: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n转录完成，耗时 {elapsed:.1f} 秒。")

    if not result:
        print("[错误] 模型返回了空结果。", file=sys.stderr)
        sys.exit(1)

    return result


def format_output(result):
    """
    将 FunASR 返回的原始结果格式化为易读的结构化数据。

    返回:
        paragraphs: 按说话人组织的段落文本
        segments:   带时间戳的逐句列表
        spk_stats:  每个说话人的统计信息
    """
    sentence_info = result[0].get("sentence_info", [])
    if not sentence_info:
        print("[警告] 未获取到句子级别的说话人信息。", file=sys.stderr)
        text = result[0].get("text", "")
        return text, [], {}

    # 按说话人组织
    spk_segments: dict[int, list[dict]] = defaultdict(list)
    segments = []

    for item in sentence_info:
        spk = item.get("spk", -1)
        # FunASR 返回的时间戳单位为毫秒，转为秒
        start_ms = item.get("start", 0)
        end_ms = item.get("end", 0)
        seg = {
            "spk": spk,
            "start": start_ms / 1000.0,
            "end": end_ms / 1000.0,
            "text": item.get("text", "").strip(),
        }
        segments.append(seg)
        spk_segments[spk].append(seg)

    # 构建按说话人分组的段落文本
    paragraphs = []
    for spk_id in sorted(spk_segments.keys()):
        segs = sorted(spk_segments[spk_id], key=lambda x: x["start"])
        full_text = "".join([s["text"] for s in segs])
        total_duration = sum(s["end"] - s["start"] for s in segs)
        paragraphs.append({
            "spk_id": spk_id,
            "spk_label": f"说话人{spk_id + 1}",
            "text": full_text,
            "segments": segs,
            "total_duration": total_duration,
            "segment_count": len(segs),
        })

    # 统计信息
    spk_stats = {
        f"说话人{s['spk_id'] + 1}": {
            "发言片段数": s["segment_count"],
            "发言总时长(秒)": round(s["total_duration"], 1),
            "首句时间": format_time(s["segments"][0]["start"]),
        }
        for s in paragraphs
    }

    return paragraphs, segments, spk_stats


def main():
    parser = argparse.ArgumentParser(
        description="FunASR 语音转文本 + 说话人分离工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python transcribe_speaker.py --audio meeting.mp3
    python transcribe_speaker.py --audio meeting.mp3 --num-speakers 3
    python transcribe_speaker.py --audio meeting.mp3 --output-json
        """,
    )
    parser.add_argument(
        "--audio", required=True,
        help="音频文件路径（支持 wav, mp3, flac, m4a 等）",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="预设说话人数量（不指定则自动估计）",
    )
    parser.add_argument(
        "--output-json", action="store_true",
        help="同时输出带时间戳的 JSON 文件",
    )

    args = parser.parse_args()
    audio_path = args.audio

    # --- 执行转录 ---
    result = transcribe_with_speakers(
        audio_path=audio_path,
        preset_spk_num=args.num_speakers,
        device="cuda:0" if _has_cuda() else "cpu",
    )

    # --- 格式化输出 ---
    paragraphs, segments, spk_stats = format_output(result)

    base_name = os.path.splitext(audio_path)[0]

    # --- 写 TXT 文件（带说话人标注） ---
    txt_path = base_name + "_speaker.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"音频文件: {os.path.basename(audio_path)}\n")
        f.write(f"检测到 {len(paragraphs)} 位说话人\n")
        f.write("=" * 70 + "\n\n")

        if isinstance(paragraphs, str):
            # 没有句子信息时的回退
            f.write(paragraphs)
        else:
            for p in paragraphs:
                f.write(f"【{p['spk_label']}】\n")
                f.write(f"(共{p['segment_count']}段发言, 约{format_time(p['total_duration'])})\n")
                f.write("-" * 50 + "\n")
                f.write(p["text"] + "\n\n")

    print(f"\n文本结果已保存至: {txt_path}")

    # --- 写 JSON 文件（可选，带详细时间戳） ---
    if args.output_json:
        json_path = base_name + "_speaker.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "audio_file": os.path.basename(audio_path),
                "speaker_count": len(paragraphs) if not isinstance(paragraphs, str) else 0,
                "speakers": [
                    {
                        "label": p["spk_label"],
                        "id": p["spk_id"],
                        "total_duration_sec": round(p["total_duration"], 1),
                        "segment_count": p["segment_count"],
                    }
                    for p in paragraphs
                ] if not isinstance(paragraphs, str) else [],
                "segments": [
                    {
                        "spk": f"说话人{s['spk'] + 1}",
                        "start": format_time(s["start"]),
                        "start_sec": round(s["start"], 2),
                        "end_sec": round(s["end"], 2),
                        "text": s["text"],
                    }
                    for s in segments
                ],
            }, f, ensure_ascii=False, indent=2)
        print(f"JSON 结果已保存至: {json_path}")

    # --- 终端总结 ---
    print("\n" + "=" * 70)
    print("说话人统计:")
    for label, stats in spk_stats.items():
        print(f"  {label}: {stats['发言片段数']} 段, "
              f"约 {stats['发言总时长(秒)']} 秒, "
              f"首句 @ {stats['首句时间']}")
    print("=" * 70)

    # --- 打印带说话人标注的逐句文本 ---
    print("\n转录全文（按时间轴）:\n")
    for seg in segments:
        spk_label = f"说话人{seg['spk'] + 1}"
        print(f"[{format_time(seg['start'])}] {spk_label}: {seg['text']}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
