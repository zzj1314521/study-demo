#!/usr/bin/env python3
"""
FunASR 语音转文本 + 说话人分离 → 对话形式输出

使用 CAM++ 声纹识别 + 谱聚类自动区分说话人，
输出按时间轴排列、合并同说话人连续片段的对话文本。

用法:
    python transcribe_dialogue.py --audio <音频文件路径>
    python transcribe_dialogue.py --audio <音频文件路径> --num-speakers 3
"""

import argparse
import json
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from identify_speaker import (
    resolve_profile,
    match_speakers_in_meeting,
    apply_labels_to_turns,
)


def format_time(seconds: float) -> str:
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def build_dialogue(sentence_info, merge_gap_sec=2.0):
    """
    将 CAM++ 的逐句标注结果合并为对话话轮。

    参数:
        sentence_info: FunASR 返回的句子列表，每项含 {text, start, end, spk}
        merge_gap_sec: 同一说话人相邻两句间隔小于此阈值则合并（秒）

    返回:
        turns: [{speaker, start, end, text, segments_count}]
    """
    if not sentence_info:
        return []

    # 按时间排序
    sorted_si = sorted(sentence_info, key=lambda x: x["start"])

    turns = []
    current = None

    for item in sorted_si:
        start_s = item["start"] / 1000.0  # ms → s
        end_s = item["end"] / 1000.0
        text = item.get("text", "").strip()
        spk = item.get("spk", -1)

        if not text:
            continue

        if current is None:
            # 第一个话轮
            current = {
                "speaker": spk,
                "start": start_s,
                "end": end_s,
                "texts": [text],
            }
        elif spk == current["speaker"] and (start_s - current["end"]) <= merge_gap_sec:
            # 同说话人且时间间隔小 → 合并
            current["end"] = end_s
            current["texts"].append(text)
        else:
            # 说话人变了或间隔太长 → 结束当前话轮，开始新话轮
            current["final_text"] = "".join(current["texts"])
            turns.append(current)
            current = {
                "speaker": spk,
                "start": start_s,
                "end": end_s,
                "texts": [text],
            }

    # 最后一个话轮
    if current is not None:
        current["final_text"] = "".join(current["texts"])
        turns.append(current)

    return turns


def main():
    parser = argparse.ArgumentParser(description="FunASR 对话转录工具")
    parser.add_argument("--audio", required=True, help="音频文件路径")
    parser.add_argument("--num-speakers", type=int, default=None, help="预设说话人数量")
    parser.add_argument("--merge-gap", type=float, default=2.0,
                        help="同说话人合并间隔阈值（秒），默认 2.0")
    parser.add_argument("--output-json", action="store_true", help="同时输出 JSON")
    parser.add_argument("--speaker-profile", default=None,
                        help="说话人声纹配置（名称或目录路径）。"
                             "如注册过「蒸馏目标人」，直接填 --speaker-profile 蒸馏目标人")
    parser.add_argument("--speaker-threshold", type=float, default=0.65,
                        help="声纹匹配阈值（默认 0.65），越高越严格")
    parser.add_argument("--device", default=None,
                        help="设备（如 cuda:7 或 cpu），默认自动检测")
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"[错误] 文件不存在: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # --- 加载模型 ---
    from funasr import AutoModel

    print("=" * 70)
    print("正在初始化模型: paraformer-zh + fsmn-vad + ct-punc + cam++")
    print("=" * 70)

    extra_kwargs = {}
    if args.num_speakers is not None:
        extra_kwargs["preset_spk_num"] = args.num_speakers

    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        device=device,
        disable_update=True,
    )

    print(f"\n设备: {device}, 合并间隔: {args.merge_gap}秒")
    print(f"音频: {audio_path}")
    print("-" * 70)

    # --- 执行推理 ---
    start_time = time.time()
    result = model.generate(input=audio_path, return_spk_res=True, **extra_kwargs)
    elapsed = time.time() - start_time

    sentence_info = result[0].get("sentence_info", [])
    if not sentence_info:
        print("[错误] 未获取到句子信息", file=sys.stderr)
        sys.exit(1)

    print(f"转录完成, 耗时 {elapsed:.0f} 秒\n")

    # --- 说话人识别（声纹匹配）---
    speaker_map = {}
    if args.speaker_profile:
        print("─" * 50)
        print(f"正在匹配说话人: {args.speaker_profile}")
        try:
            profile = resolve_profile(args.speaker_profile)
            match_result = match_speakers_in_meeting(
                audio_path=audio_path,
                sentence_info=sentence_info,
                profiles=[profile],
                threshold=args.speaker_threshold,
                device=device,
            )
            speaker_map = match_result["speaker_map"]
            if match_result["matches"]:
                m = match_result["matches"][0]
                print(f"结果: 说话人{m['spk_id']} → 「{m['name']}」(相似度 {m['similarity']})")
            else:
                print("结果: 未匹配到目标说话人")
        except Exception as e:
            print(f"[警告] 说话人识别失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        print("─" * 50 + "\n")

    # --- 构建对话 ---
    turns = build_dialogue(sentence_info, merge_gap_sec=args.merge_gap)

    # --- 应用声纹标签 ---
    if speaker_map:
        turns = apply_labels_to_turns(turns, speaker_map)

    # --- 统计（以字符串标签为准）---
    spk_labels = sorted(set(t["speaker"] for t in turns))
    spk_stats = {}
    for label in spk_labels:
        spk_turns = [t for t in turns if t["speaker"] == label]
        total_sec = sum(t["end"] - t["start"] for t in spk_turns)
        spk_stats[label] = {"turns": len(spk_turns), "duration": total_sec}

    # --- 输出文件 ---
    base = os.path.splitext(audio_path)[0]
    txt_path = base + "_dialogue.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"音频: {os.path.basename(audio_path)}\n")
        f.write(f"说话人数: {len(spk_labels)}, 对话话轮: {len(turns)}\n")
        f.write("=" * 70 + "\n\n")

        for t in turns:
            label = str(t["speaker"])  # 已经是字符串标签
            ts = format_time(t["start"])
            f.write(f"[{ts}] {label}: {t['final_text']}\n\n")

        f.write("\n--- 说话人统计 ---\n")
        for label in spk_labels:
            dur = format_time(spk_stats[label]["duration"])
            f.write(f"  {label}: {spk_stats[label]['turns']} 个话轮, 约 {dur}\n")

    print(f"对话文本已保存: {txt_path}")

    # --- JSON ---
    if args.output_json:
        json_path = base + "_dialogue.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "audio": os.path.basename(audio_path),
                "speakers": len(spk_labels),
                "total_turns": len(turns),
                "turns": [
                    {
                        "speaker": str(t["speaker"]),
                        "start_sec": round(t["start"], 2),
                        "end_sec": round(t["end"], 2),
                        "time": format_time(t["start"]),
                        "text": t["final_text"],
                    }
                    for t in turns
                ],
            }, f, ensure_ascii=False, indent=2)
        print(f"JSON 已保存: {json_path}")

    # --- 终端预览前 20 个话轮 ---
    print("\n" + "=" * 70)
    print(f"对话预览（共 {len(turns)} 话轮）:\n")
    for t in turns[:20]:
        label = str(t["speaker"])
        ts = format_time(t["start"])
        print(f"[{ts}] {label}: {t['final_text']}")
    if len(turns) > 20:
        print(f"\n  ... (省略 {len(turns) - 20} 话轮) ...")
    print("=" * 70)


if __name__ == "__main__":
    main()
