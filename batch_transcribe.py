#!/usr/bin/env python3
"""
批量音频转写脚本 — 对话形式输出

遍历指定目录下所有 MP3 文件，逐一进行语音识别 + 说话人分离，
输出带时间轴标注的对话文本。

用法:
    python batch_transcribe.py --dir <音频目录>
    python batch_transcribe.py --dir <音频目录> --num-speakers 3
    python batch_transcribe.py --dir <音频目录> --skip-existing   # 跳过已有输出
    python batch_transcribe.py --dir <音频目录> --merge-gap 1.5

输出:
    每个 MP3 文件生成同名的 _dialogue.txt 和 _dialogue.json
    + 批量汇总文件 _batch_summary.txt
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

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
    """将逐句标注结果合并为对话话轮。"""
    if not sentence_info:
        return []

    sorted_si = sorted(sentence_info, key=lambda x: x["start"])
    turns = []
    current = None

    for item in sorted_si:
        start_s = item["start"] / 1000.0
        end_s = item["end"] / 1000.0
        text = item.get("text", "").strip()
        spk = item.get("spk", -1)

        if not text:
            continue

        if current is None:
            current = {"speaker": spk, "start": start_s, "end": end_s, "texts": [text]}
        elif spk == current["speaker"] and (start_s - current["end"]) <= merge_gap_sec:
            current["end"] = end_s
            current["texts"].append(text)
        else:
            current["final_text"] = "".join(current["texts"])
            turns.append(current)
            current = {"speaker": spk, "start": start_s, "end": end_s, "texts": [text]}

    if current is not None:
        current["final_text"] = "".join(current["texts"])
        turns.append(current)

    return turns


def process_file(model, mp3_path: str, num_speakers: int, merge_gap: float,
                 output_json: bool = False, profile: dict = None,
                 speaker_threshold: float = 0.65,
                 txt_dir: str = None, json_dir: str = None,
                 device: str = None):
    """
    处理单个 MP3 文件，返回结果摘要。

    参数:
        profile: 已加载的说话人配置 {"name": ..., "embedding": ...}，为 None 则不匹配
        speaker_threshold: 声纹匹配阈值
        txt_dir: TXT 输出目录（默认与音频同目录）
        json_dir: JSON 输出目录（默认与音频同目录）
        device: 设备（如 "cuda:7"），传给声纹匹配

    返回:
        dict: {status, turns, speakers, duration_sec, error, speaker_map}
    """
    try:
        result = model.generate(
            input=mp3_path,
            return_spk_res=True,
            preset_spk_num=num_speakers if num_speakers else None,
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}

    sentence_info = result[0].get("sentence_info", [])
    if not sentence_info:
        return {"status": "error", "error": "无句子信息"}

    # --- 声纹匹配 ---
    speaker_map = {}
    if profile:
        try:
            match_result = match_speakers_in_meeting(
                audio_path=mp3_path,
                sentence_info=sentence_info,
                profiles=[profile],
                threshold=speaker_threshold,
                device=device,
            )
            speaker_map = match_result["speaker_map"]
        except Exception:
            pass  # 匹配失败不影响继续

    turns = build_dialogue(sentence_info, merge_gap_sec=merge_gap)
    if not turns:
        return {"status": "error", "error": "未生成话轮"}

    # 应用声纹标签
    if speaker_map:
        turns = apply_labels_to_turns(turns, speaker_map)

    spk_labels = sorted(set(t["speaker"] for t in turns))
    total_dur = sum(t["end"] - t["start"] for t in turns)

    # --- 写 TXT ---
    fname_base = os.path.splitext(os.path.basename(mp3_path))[0]
    if txt_dir:
        os.makedirs(txt_dir, exist_ok=True)
        txt_path = os.path.join(txt_dir, fname_base + "_dialogue.txt")
    else:
        txt_path = os.path.splitext(mp3_path)[0] + "_dialogue.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"音频: {os.path.basename(mp3_path)}\n")
        f.write(f"说话人数: {len(spk_labels)}, 话轮: {len(turns)}\n")
        f.write("=" * 70 + "\n\n")
        for t in turns:
            label = str(t["speaker"])
            ts = format_time(t["start"])
            f.write(f"[{ts}] {label}: {t['final_text']}\n\n")
        f.write("\n--- 统计 ---\n")
        for label in spk_labels:
            spk_turns = [t for t in turns if t["speaker"] == label]
            sec = sum(t["end"] - t["start"] for t in spk_turns)
            f.write(f"  {label}: {len(spk_turns)} 话轮, 约{format_time(sec)}\n")

    # --- 写 JSON（可选）---
    if output_json:
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
            json_path = os.path.join(json_dir, fname_base + "_dialogue.json")
        else:
            json_path = os.path.splitext(mp3_path)[0] + "_dialogue.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "audio": os.path.basename(mp3_path),
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

    return {
        "status": "ok",
        "turns": len(turns),
        "speakers": len(spk_labels),
        "duration_sec": total_dur,
        "speaker_map": speaker_map,
    }


def find_mp3_files(directory: str):
    """递归查找目录下所有 MP3 文件，按文件名排序。"""
    mp3_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".mp3"):
                mp3_files.append(os.path.join(root, f))
    return sorted(mp3_files)


def main():
    parser = argparse.ArgumentParser(
        description="批量音频转写 — 对话形式输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", default=None, help="包含 MP3 文件的目录")
    parser.add_argument("--files", default=None, help="文件列表（一行一个 MP3 路径，与 --dir 二选一）")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="预设说话人数量（不指定则自动估计）")
    parser.add_argument("--merge-gap", type=float, default=2.0,
                        help="同说话人合并阈值（秒），默认 2.0")
    parser.add_argument("--output-json", action="store_true",
                        help="同时输出 JSON 文件")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有 _dialogue.txt 的文件")
    parser.add_argument("--speaker-profile", default=None,
                        help="说话人声纹配置名称或目录（如「蒸馏目标人」）")
    parser.add_argument("--speaker-threshold", type=float, default=0.65,
                        help="声纹匹配阈值（默认 0.65）")
    parser.add_argument("--device", default=None,
                        help="设备（如 cuda:7 或 cpu），默认自动检测")
    parser.add_argument("--txt-dir", default=None,
                        help="TXT 输出目录（默认与音频同目录）")
    parser.add_argument("--json-dir", default=None,
                        help="JSON 输出目录（默认与音频同目录）")
    args = parser.parse_args()

    # --- 解析声纹配置 ---
    profile = None
    if args.speaker_profile:
        print(f"正在加载说话人声纹: {args.speaker_profile}")
        try:
            profile = resolve_profile(args.speaker_profile)
            print(f"  已加载「{profile['name']}」的声纹\n")
        except Exception as e:
            print(f"[错误] 无法加载声纹: {e}", file=sys.stderr)
            sys.exit(1)

    # --- 获取 MP3 列表 ---
    if args.files:
        # 从文件列表读取
        if not os.path.isfile(args.files):
            print(f"[错误] 文件列表不存在: {args.files}", file=sys.stderr)
            sys.exit(1)
        with open(args.files, "r", encoding="utf-8") as f:
            mp3_files = [line.strip() for line in f if line.strip()]
        directory = os.path.dirname(mp3_files[0]) if mp3_files else "."
        print(f"从列表读取 {len(mp3_files)} 个文件")
    else:
        if not args.dir:
            print("[错误] 需要 --dir 或 --files 参数", file=sys.stderr)
            sys.exit(1)
        directory = os.path.abspath(args.dir)
        if not os.path.isdir(directory):
            print(f"[错误] 目录不存在: {directory}", file=sys.stderr)
            sys.exit(1)
        mp3_files = find_mp3_files(directory)
        if not mp3_files:
            print(f"[错误] 目录下未找到 MP3 文件: {directory}", file=sys.stderr)
            sys.exit(1)
        print(f"找到 {len(mp3_files)} 个 MP3 文件")

    print(f"目录: {directory}")
    print(f"说话人数: {'自动' if args.num_speakers is None else args.num_speakers}")
    print(f"跳过已有: {'是' if args.skip_existing else '否'}")
    print()

    # --- 过滤已处理的 ---
    pending = []
    skipped = 0
    for mp3 in mp3_files:
        fname_base = os.path.splitext(os.path.basename(mp3))[0]
        if args.txt_dir:
            txt_path = os.path.join(args.txt_dir, fname_base + "_dialogue.txt")
        else:
            txt_path = os.path.splitext(mp3)[0] + "_dialogue.txt"
        if args.skip_existing and os.path.exists(txt_path):
            skipped += 1
            continue
        pending.append(mp3)

    if skipped > 0:
        print(f"跳过 {skipped} 个已处理文件")
    print(f"待处理: {len(pending)} 个\n")

    if not pending:
        print("没有需要处理的文件。")
        sys.exit(0)

    # --- 加载模型（只加载一次）---
    from funasr import AutoModel
    import torch

    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print("正在初始化模型（paraformer-zh + fsmn-vad + ct-punc + cam++）...")

    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        device=device,
        disable_update=True,
    )
    print("模型就绪。\n")

    # --- 逐文件处理 ---
    results = []
    total_start = time.time()

    for i, mp3 in enumerate(pending, 1):
        fname = os.path.basename(mp3)
        fsize_mb = os.path.getsize(mp3) / (1024 * 1024)

        print(f"[{i}/{len(pending)}] {fname} ({fsize_mb:.1f} MB) ... ", end="", flush=True)

        t_start = time.time()
        res = process_file(
            model, mp3,
            num_speakers=args.num_speakers,
            merge_gap=args.merge_gap,
            output_json=args.output_json,
            profile=profile,
            speaker_threshold=args.speaker_threshold,
            txt_dir=args.txt_dir,
            json_dir=args.json_dir,
            device=device,
        )
        elapsed = time.time() - t_start

        if res["status"] == "ok":
            print(f"✓ {res['turns']} 话轮, "
                  f"{res['speakers']} 人, "
                  f"{elapsed:.0f}s")
            res["file"] = fname
            res["elapsed"] = elapsed
            results.append(res)
        else:
            print(f"✗ 失败: {res['error']}")
            results.append({"file": fname, "status": "error",
                           "error": res["error"], "elapsed": elapsed})

    total_elapsed = time.time() - total_start

    # --- 汇总 ---
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")

    summary_path = os.path.join(directory, "_batch_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"批量转写汇总\n")
        f.write(f"目录: {directory}\n")
        f.write(f"总文件: {len(mp3_files)}, 成功: {ok_count}, 失败: {err_count}\n")
        f.write(f"总耗时: {format_time(total_elapsed)}\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            if r["status"] == "ok":
                f.write(f"  ✓ {r['file']} — {r['turns']}话轮, {r['speakers']}人, {r['elapsed']:.0f}s\n")
            else:
                f.write(f"  ✗ {r['file']} — {r['error']}\n")

    print(f"\n{'=' * 70}")
    print(f"完成: 成功 {ok_count}/{len(pending)}, 失败 {err_count}")
    print(f"总耗时: {format_time(total_elapsed)}")
    print(f"汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
