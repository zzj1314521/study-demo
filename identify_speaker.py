#!/usr/bin/env python3
"""
说话人识别模块 — 基于 CAM++ 声纹比对（路由 A：后处理切割）

通过参考音频注册目标说话人声纹，然后在任意会议中自动识别该说话人，
将无意义的「说话人0/1/2」标签替换为自定义名称（如「蒸馏目标人」）。

用法:
    # 注册参考说话人
    python identify_speaker.py register --audio <参考音频.mp3> --name 蒸馏目标人

    # 列出已注册的说话人
    python identify_speaker.py list

    # 对单个会议 JSON 做说话人识别
    python identify_speaker.py identify --audio <会议.mp3> --json <dialogue.json> --profile 蒸馏目标人

    # 删除某个配置
    python identify_speaker.py delete --name 蒸馏目标人

也可作为模块导入:
    from identify_speaker import (
        load_cam_model, register_speaker, load_profile, load_all_profiles,
        match_speakers_in_meeting, apply_labels_to_turns,
    )
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── 路径配置 ─────────────────────────────────────────────
DEFAULT_PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profiles")

# ─── 工具函数 ─────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，范围 [-1, 1]，越高越相似。"""
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def check_ffmpeg():
    """确认 ffmpeg 可用，不可用则抛出。"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("未找到 ffmpeg，请安装后添加到 PATH。https://www.gyan.dev/ffmpeg/builds/")


# ─── CAM++ 模型加载 ────────────────────────────────────────

_cam_model = None  # 模块级缓存


def load_cam_model(device: str = None):
    """加载 CAM++ 声纹模型（模块级单例，只加载一次）。"""
    global _cam_model
    if _cam_model is None:
        from funasr import AutoModel
        kwargs = {"model": "cam++", "disable_update": True}
        if device:
            kwargs["device"] = device
        _cam_model = AutoModel(**kwargs)
    return _cam_model


def extract_embedding(model, audio_path: str) -> np.ndarray:
    """
    从音频文件提取 CAM++ 声纹向量 (shape: 192,)。
    如果音频包含多段语音，返回多段的平均声纹。
    """
    import torch as _torch
    res = model.generate(input=audio_path)
    emb = res[0].get("spk_embedding")
    if emb is None:
        raise RuntimeError(f"未能从音频提取声纹: {audio_path}")
    # 处理 GPU tensor（FunASR 可能返回 CUDA tensor）
    if isinstance(emb, _torch.Tensor):
        emb = emb.detach().cpu().numpy()
    else:
        emb = np.asarray(emb)
    # 可能是 (N, 192) 或 (192,)
    if emb.ndim == 2:
        emb = emb.mean(axis=0)
    return emb.flatten()


# ─── 注册参考说话人 ────────────────────────────────────────

def register_speaker(audio_path: str, name: str, profiles_dir: str = None) -> dict:
    """
    从参考音频注册说话人（首个声纹）。

    参数:
        audio_path: 参考音频路径
        name: 说话人名称（如「蒸馏目标人」）
        profiles_dir: profiles 存储目录

    返回:
        {"profile_dir": ..., "embedding": np.ndarray, "name": ...}
    """
    if profiles_dir is None:
        profiles_dir = DEFAULT_PROFILES_DIR

    profile_dir = os.path.join(profiles_dir, name)
    ensure_dir(profile_dir)

    print(f"正在加载 CAM++ 模型...")
    model = load_cam_model()

    print(f"正在从参考音频提取声纹: {audio_path}")
    emb = extract_embedding(model, audio_path)
    print(f"  声纹维度: {emb.shape[0]}")

    # 保存主 embedding
    emb_path = os.path.join(profile_dir, "embedding.npy")
    np.save(emb_path, emb)

    # 保存元信息
    info = {
        "name": name,
        "source_audios": [os.path.basename(audio_path)],
        "embedding_dim": int(emb.shape[0]),
        "embedding_count": 1,
        "created": datetime.now().isoformat(),
    }
    info_path = os.path.join(profile_dir, "info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 说话人「{name}」已注册")
    print(f"  配置目录: {profile_dir}")
    print(f"  声纹文件: {emb_path}")

    return {"profile_dir": profile_dir, "embedding": emb, "name": name}


# ─── 加载 Profile ──────────────────────────────────────────

def add_reference(audio_path: str, name: str, profiles_dir: str = None) -> dict:
    """
    为已有说话人追加一条参考声纹（用于覆盖不同的录音环境）。

    参数:
        audio_path: 新增参考音频路径
        name: 已注册的说话人名称
        profiles_dir: profiles 存储目录
    """
    if profiles_dir is None:
        profiles_dir = DEFAULT_PROFILES_DIR

    profile_dir = os.path.join(profiles_dir, name)
    if not os.path.isdir(profile_dir):
        raise FileNotFoundError(f"未找到说话人「{name}」，请先用 register 命令注册")

    model = load_cam_model()
    print(f"正在从新增参考音频提取声纹: {audio_path}")
    emb = extract_embedding(model, audio_path)

    # 存入 embeddings/ 子目录
    emb_dir = os.path.join(profile_dir, "embeddings")
    ensure_dir(emb_dir)
    existing = sorted([f for f in os.listdir(emb_dir) if f.endswith('.npy')])
    next_idx = len(existing) + 1
    emb_path = os.path.join(emb_dir, f"{next_idx:03d}.npy")
    np.save(emb_path, emb)

    # 更新元信息
    info_path = os.path.join(profile_dir, "info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    info["embedding_count"] = 1 + len(existing)
    audios = info.get("source_audios", [info.get("source_audio", "?")])
    audios.append(os.path.basename(audio_path))
    info["source_audios"] = audios
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 已为「{name}」追加第 {info['embedding_count']} 条参考声纹")
    print(f"  声纹文件: {emb_path}")
    return {"profile_dir": profile_dir, "embedding": emb, "name": name}


def load_profile(profile_dir: str) -> dict:
    """
    加载单个说话人 profile（支持多条参考声纹）。

    参数:
        profile_dir: profiles 目录下的某个说话人目录

    返回:
        {"name": str, "embeddings": [np.ndarray, ...], "profile_dir": str}
    """
    info_path = os.path.join(profile_dir, "info.json")
    name = os.path.basename(profile_dir)
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        name = info.get("name", name)

    embeddings = []

    # 主声纹
    emb_path = os.path.join(profile_dir, "embedding.npy")
    if os.path.exists(emb_path):
        embeddings.append(np.load(emb_path))

    # 附加声纹
    emb_dir = os.path.join(profile_dir, "embeddings")
    if os.path.isdir(emb_dir):
        for fname in sorted(os.listdir(emb_dir)):
            if fname.endswith(".npy"):
                embeddings.append(np.load(os.path.join(emb_dir, fname)))

    if not embeddings:
        raise FileNotFoundError(f"声纹文件不存在: {profile_dir}")

    return {"name": name, "embeddings": embeddings, "profile_dir": profile_dir}


def load_all_profiles(profiles_dir: str = None) -> list:
    """
    加载所有已注册的说话人 profile。

    返回:
        [{"name": ..., "embedding": ..., "profile_dir": ...}, ...]
    """
    if profiles_dir is None:
        profiles_dir = DEFAULT_PROFILES_DIR
    if not os.path.isdir(profiles_dir):
        return []

    profiles = []
    for entry in sorted(os.listdir(profiles_dir)):
        entry_path = os.path.join(profiles_dir, entry)
        if os.path.isdir(entry_path):
            try:
                profiles.append(load_profile(entry_path))
            except Exception as e:
                print(f"[警告] 跳过损坏的 profile '{entry}': {e}", file=sys.stderr)
    return profiles


def resolve_profile(profile_spec: str, profiles_dir: str = None) -> dict:
    """
    解析 profile 参数（可以是名称或路径）。

    返回:
        {"name": str, "embedding": np.ndarray, "profile_dir": str}
    """
    if profiles_dir is None:
        profiles_dir = DEFAULT_PROFILES_DIR

    # 1. 先当作完整路径
    if os.path.isdir(profile_spec):
        return load_profile(profile_spec)

    # 2. 在 profiles_dir 下寻找同名目录
    candidate = os.path.join(profiles_dir, profile_spec)
    if os.path.isdir(candidate):
        return load_profile(candidate)

    # 3. 模糊匹配
    all_profiles = load_all_profiles(profiles_dir)
    matches = [p for p in all_profiles if profile_spec in p["name"]]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        names = ", ".join(p["name"] for p in matches)
        raise ValueError(f"找到多个匹配的 profile: {names}，请指定更精确的名称")
    else:
        raise FileNotFoundError(f"未找到 profile: {profile_spec}（在 {profiles_dir} 下）")


# ─── 会议中的说话人匹配 ────────────────────────────────────

_cut_embed_error_reported = False  # 每个文件只报一次错，避免刷屏

def _cut_and_embed(cam_model, audio_path: str, start_s: float, end_s: float,
                   work_dir: str, seg_id: str) -> np.ndarray:
    """裁剪一段音频并提取 CAM++ 声纹，失败返回 None。"""
    global _cut_embed_error_reported
    seg_path = os.path.join(work_dir, f"{seg_id}.wav")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", audio_path,
        "-t", f"{end_s - start_s:.3f}",
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le",
        seg_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.getsize(seg_path) > 0:
            emb = extract_embedding(cam_model, seg_path)
            return emb
        elif not _cut_embed_error_reported:
            _cut_embed_error_reported = True
            print(f"  [诊断] ffmpeg 输出文件为空: {' '.join(cmd)}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        if not _cut_embed_error_reported:
            _cut_embed_error_reported = True
            print(f"  [诊断] ffmpeg 失败: {e.stderr.decode()[:300] if e.stderr else e}", file=sys.stderr)
            print(f"  [诊断] 源音频: {audio_path}", file=sys.stderr)
            print(f"  [诊断] 命令: {' '.join(cmd)}", file=sys.stderr)
    except Exception as e:
        if not _cut_embed_error_reported:
            _cut_embed_error_reported = True
            print(f"  [诊断] CAM++ 提取失败: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"  [诊断] 片段: {seg_path}", file=sys.stderr)
    return None


def _sample_segments(sentence_info: list, spk_id: int,
                     min_seg_sec: float = 3.0,
                     max_samples: int = 15) -> list:
    """
    为指定说话人从全时间线均匀采样片段。

    返回: [(start_s, end_s, dur, text), ...]  按时间排序
    """
    segs = []
    for item in sentence_info:
        if item["spk"] != spk_id:
            continue
        dur = (item["end"] - item["start"]) / 1000.0
        if dur >= min_seg_sec and item.get("text", "").strip():
            segs.append((
                item["start"] / 1000.0,
                item["end"] / 1000.0,
                dur,
                item.get("text", ""),
            ))

    segs.sort(key=lambda x: x[0])  # 按时间排序

    if len(segs) <= max_samples:
        return segs

    # 均匀采样：从全时间线取 max_samples 个
    step = len(segs) / max_samples
    sampled = []
    for i in range(max_samples):
        idx = min(int(i * step), len(segs) - 1)
        sampled.append(segs[idx])

    # 去重（相邻采样可能落到同一个段）
    unique = []
    for s in sampled:
        if not unique or s != unique[-1]:
            unique.append(s)
    return unique


def _route_a_verify(
    cam_model,
    audio_path: str,
    sentence_info: list,
    spk_id: int,
    ref_embs: list,
    work_dir: str,
) -> tuple:
    """
    Route A 备用验证：对灰度区间的说话人做更深度的声纹提取。

    与主路径的区别：
      - 最多 40 段采样（主路径 15 段）
      - 最低 1.5 秒（主路径 3 秒）
      - top-5 平均（主路径 top-3）

    返回: (avg_sim, voting, total) 或 (0.0, 0, 0)
    """
    segs = _sample_segments(sentence_info, spk_id,
                            min_seg_sec=1.5, max_samples=40)
    if not segs:
        return 0.0, 0, 0

    seg_sims = []
    success = 0
    for ss, es, dur, text in segs:
        seg_id = f"routeA_spk{spk_id}_{ss:.0f}"
        emb = _cut_and_embed(cam_model, audio_path, ss, es, work_dir, seg_id)
        if emb is not None:
            success += 1
            max_sim = max(cosine_similarity(emb, ref_emb) for ref_emb in ref_embs)
            seg_sims.append(max_sim)

    if not seg_sims:
        return 0.0, 0, 0

    seg_sims.sort(reverse=True)
    quality_sims = [s for s in seg_sims if s > 0.40]  # Route A 质量底线更低
    top_k = quality_sims[:5]  # top-5 平均
    avg_sim = float(np.mean(top_k)) if top_k else 0.0
    voting = len(quality_sims)

    print(f"    [Route A] 说话人{spk_id}: 采样{len(segs)}段 → "
          f"提取{success}段, avg={avg_sim:.4f} "
          f"(best={seg_sims[0]:.4f}, voting={voting}/{len(seg_sims)}段>0.40)")

    return avg_sim, voting, len(seg_sims)


def match_speakers_in_meeting(
    audio_path: str,
    sentence_info: list,
    profiles: list,
    threshold: float = 0.65,
    work_dir: str = None,
    device: str = None,
    gray_zone_low: float = 0.50,
    route_a_threshold: float = 0.55,
) -> dict:
    """
    在单场会议中匹配已注册的说话人（智能采样 + 多参考融合 + Route A 备用）。

    算法：
      1. 对每种 spk，从全时间线均匀采样最多 15 段
      2. 每段独立提取 CAM++ 声纹
      3. 逐个与参考声纹比对
      4. 取相似度最高的 top-3 段平均 → 该 spk 的最终相似度
      5. 多参考取 max 相似度
      6. 超过阈值则匹配
      7. Route A: 若 best_sim 在 [gray_zone_low, threshold) 区间，
         对最佳候选说话人做更深采样（40段/1.5s/top-5），
         若 Route A 相似度 ≥ route_a_threshold 则匹配

    参数:
        audio_path: 会议音频路径
        sentence_info: FunASR 返回的句子列表（含 spk 标注）
        profiles: 已注册的说话人列表 [{"name": ..., "embeddings": [np.ndarray, ...]}, ...]
        threshold: 主路径余弦相似度阈值（默认 0.65）
        work_dir: 临时目录
        device: 设备（如 "cuda:7" 或 "cpu"），为 None 则自动选择
        gray_zone_low: 灰度区间下限（默认 0.50），低于此值不触发 Route A
        route_a_threshold: Route A 判定阈值（默认 0.55）

    返回:
        {speaker_map, matches, unmatched_spk_ids}
    """
    global _cut_embed_error_reported
    _cut_embed_error_reported = False  # 每个文件重新诊断

    check_ffmpeg()

    own_tmp = work_dir is None
    if own_tmp:
        work_dir = tempfile.mkdtemp(prefix="speaker_id_")

    try:
        cam = load_cam_model(device=device)
        all_spk_ids = sorted(set(item["spk"] for item in sentence_info))

        if not all_spk_ids:
            return _empty_result(set())

        # ─── 逐 spk 采样 + 提取声纹 ───
        spk_seg_embs = {}  # {spk_id: [(start_s, embedding, text_preview), ...]}

        for spk_id in all_spk_ids:
            segs = _sample_segments(sentence_info, spk_id)
            if not segs:
                continue

            results = []
            for ss, es, dur, text in segs:
                seg_id = f"spk{spk_id}_{ss:.0f}"
                emb = _cut_and_embed(cam, audio_path, ss, es, work_dir, seg_id)
                if emb is not None:
                    results.append((ss, emb, text[:40]))

            if results:
                spk_seg_embs[spk_id] = results
                print(f"    说话人{spk_id}: {len(segs)}段采样 → {len(results)}段声纹提取成功")

        if not spk_seg_embs:
            print("  [警告] 未能提取任何说话人声纹")
            return _empty_result(set(all_spk_ids))

        # ─── 逐 profile 匹配（多参考取 max）───
        speaker_map = {}
        matches = []

        for profile in profiles:
            ref_name = profile["name"]
            ref_embs = profile["embeddings"]  # list of np.ndarray

            best_spk = None
            best_sim = -1.0

            for spk_id, seg_results in spk_seg_embs.items():
                # 对每个片段，计算与所有参考的最大相似度
                seg_sims = []
                for _ss, seg_emb, _text in seg_results:
                    # 多参考取 max
                    max_sim = max(cosine_similarity(seg_emb, ref_emb) for ref_emb in ref_embs)
                    seg_sims.append(max_sim)

                # 质量过滤：只保留 > 质量底线的段，取 top-3 平均
                seg_sims.sort(reverse=True)
                quality_sims = [s for s in seg_sims if s > 0.45]
                top_k = quality_sims[:3]
                avg_sim = float(np.mean(top_k)) if top_k else 0.0
                voting = len(quality_sims)

                print(f"    {ref_name} ←→ 说话人{spk_id}: {avg_sim:.4f} "
                      f"(best={seg_sims[0]:.4f}, voting={voting}/{len(seg_sims)}段>0.45)")

                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_spk = spk_id

            if best_spk is not None and best_sim >= threshold:
                # ─── 主路径匹配成功 ───
                speaker_map[best_spk] = ref_name
                matches.append({
                    "spk_id": best_spk,
                    "name": ref_name,
                    "similarity": round(best_sim, 4),
                    "source": "primary",
                })
                print(f"  ✓ 说话人{best_spk} 匹配为「{ref_name}」(相似度 {best_sim:.4f})")

            elif best_spk is not None and best_sim >= gray_zone_low:
                # ─── Route A 备用验证 ───
                print(f"  → 触发 Route A: 说话人{best_spk} 相似度 {best_sim:.4f} "
                      f"在灰度区间 [{gray_zone_low}, {threshold})")
                route_a_sim, route_a_voting, route_a_total = _route_a_verify(
                    cam, audio_path, sentence_info, best_spk,
                    ref_embs, work_dir,
                )
                if route_a_sim >= route_a_threshold:
                    speaker_map[best_spk] = ref_name
                    matches.append({
                        "spk_id": best_spk,
                        "name": ref_name,
                        "similarity": round(route_a_sim, 4),
                        "source": "route_a",
                    })
                    print(f"  ✓ Route A 确认: 说话人{best_spk} 匹配为「{ref_name}」"
                          f"(相似度 {route_a_sim:.4f}, voting={route_a_voting})")
                else:
                    print(f"  ✗ Route A 未通过: 说话人{best_spk} "
                          f"(相似度 {route_a_sim:.4f} < {route_a_threshold})")

            else:
                print(f"  ✗ 未找到「{ref_name}」的匹配（最高相似度 {best_sim:.4f} < {threshold}）")

        # ─── 未匹配的保持默认名称 ───
        matched_ids = set(speaker_map.keys())
        unmatched = sorted(set(all_spk_ids) - matched_ids)
        next_num = 1
        for spk_id in sorted(all_spk_ids):
            if spk_id not in speaker_map:
                while f"说话人{next_num}" in speaker_map.values():
                    next_num += 1
                speaker_map[spk_id] = f"说话人{next_num}"
                next_num += 1

        return {
            "speaker_map": speaker_map,
            "matches": matches,
            "unmatched_spk_ids": unmatched,
        }

    finally:
        if own_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)


def _empty_result(all_spk_ids: set) -> dict:
    """无匹配时的默认结果。"""
    speaker_map = {}
    for i, spk_id in enumerate(sorted(all_spk_ids), 1):
        speaker_map[spk_id] = f"说话人{i}"
    return {
        "speaker_map": speaker_map,
        "matches": [],
        "unmatched_spk_ids": sorted(all_spk_ids),
    }


def apply_labels_to_turns(turns: list, speaker_map: dict) -> list:
    """
    将话轮列表中的 spk 数字标签替换为自定义名称。

    参数:
        turns: build_dialogue 产出的列表 [{speaker, start, end, final_text}, ...]
        speaker_map: {原始spk_id: "显示名称"}

    返回:
        更新后的 turns（原地修改 + 返回）
    """
    for t in turns:
        old_spk = t["speaker"]
        t["speaker"] = speaker_map.get(old_spk, f"说话人{old_spk + 1}")
    return turns


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def cmd_register(args):
    """注册参考说话人。"""
    if not os.path.exists(args.audio):
        print(f"[错误] 音频文件不存在: {args.audio}", file=sys.stderr)
        sys.exit(1)

    register_speaker(
        audio_path=args.audio,
        name=args.name,
        profiles_dir=args.profiles_dir,
    )


def cmd_list(args):
    """列出已注册的说话人。"""
    profiles = load_all_profiles(args.profiles_dir)
    if not profiles:
        print("(暂无已注册的说话人)")
        print(f"配置目录: {args.profiles_dir}")
        return

    print(f"已注册说话人 ({len(profiles)}):")
    print("-" * 50)
    for p in profiles:
        count = len(p["embeddings"])
        dim = p["embeddings"][0].shape[0]
        sources = ""
        info_path = os.path.join(p["profile_dir"], "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            audios = info.get("source_audios", [info.get("source_audio", "")])
            sources = ", ".join(audios)
        print(f"  ● {p['name']} ({count}条参考)")
        print(f"    来源: {sources}, 维度: {dim}")


def cmd_identify(args):
    """对单个会议执行说话人识别。"""
    if not os.path.exists(args.audio):
        print(f"[错误] 音频文件不存在: {args.audio}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.json):
        print(f"[错误] JSON 文件不存在: {args.json}", file=sys.stderr)
        sys.exit(1)

    # 加载对话 JSON
    with open(args.json, "r", encoding="utf-8") as f:
        dialogue_data = json.load(f)

    # 从 JSON 还原 sentence_info（需要原始 sentence_info）
    # 注意：dialogue.json 里只有 turns，没有原始 sentence_info。
    # 所以 identify 命令需要 --sentence-info 而非 --json。
    # 重新设计：
    print("[提示] identify 命令需要原始 sentence_info，请使用 --sentence-info 参数")
    print("       建议直接使用 transcribe_dialogue.py --speaker-profile 参数代替。")
    sys.exit(1)


def cmd_add_reference(args):
    """为已有说话人追加参考声纹。"""
    if not os.path.exists(args.audio):
        print(f"[错误] 音频文件不存在: {args.audio}", file=sys.stderr)
        sys.exit(1)
    add_reference(audio_path=args.audio, name=args.name, profiles_dir=args.profiles_dir)


def cmd_delete(args):
    """删除已注册的说话人。"""
    profile_dir = os.path.join(args.profiles_dir, args.name)
    if not os.path.isdir(profile_dir):
        print(f"[错误] 未找到说话人: {args.name}", file=sys.stderr)
        sys.exit(1)
    shutil.rmtree(profile_dir)
    print(f"已删除说话人「{args.name}」")


def main():
    parser = argparse.ArgumentParser(
        description="说话人声纹注册与识别（基于 CAM++）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # --- register ---
    p_reg = sub.add_parser("register", help="注册参考说话人")
    p_reg.add_argument("--audio", required=True, help="参考音频路径")
    p_reg.add_argument("--name", required=True, help="说话人名称（如「蒸馏目标人」）")
    p_reg.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR, help="Profile 存储目录")

    # --- list ---
    p_list = sub.add_parser("list", help="列出已注册说话人")
    p_list.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR, help="Profile 存储目录")

    # --- identify ---
    p_ident = sub.add_parser("identify", help="识别会议中的说话人（实验性，建议用 transcribe_dialogue.py --speaker-profile）")
    p_ident.add_argument("--audio", required=True, help="会议音频路径")
    p_ident.add_argument("--json", required=True, help="dialogue.json 文件")
    p_ident.add_argument("--sentence-info", default=None, help="sentence_info JSON 文件路径（如有）")
    p_ident.add_argument("--profile", required=True, help="要匹配的说话人名称或目录")
    p_ident.add_argument("--threshold", type=float, default=0.65, help="相似度阈值（默认 0.65）")
    p_ident.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR, help="Profile 存储目录")

    # --- add-reference ---
    p_add = sub.add_parser("add-reference", help="为已有说话人追加参考声纹")
    p_add.add_argument("--audio", required=True, help="新增参考音频路径")
    p_add.add_argument("--name", required=True, help="已有说话人名称")
    p_add.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR, help="Profile 存储目录")

    # --- delete ---
    p_del = sub.add_parser("delete", help="删除已注册说话人")
    p_del.add_argument("--name", required=True, help="要删除的说话人名称")
    p_del.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR, help="Profile 存储目录")

    args = parser.parse_args()

    if args.command == "register":
        cmd_register(args)
    elif args.command == "add-reference":
        cmd_add_reference(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "identify":
        cmd_identify(args)
    elif args.command == "delete":
        cmd_delete(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
