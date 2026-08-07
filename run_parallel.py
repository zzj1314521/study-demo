#!/usr/bin/env python3
"""将批量转录拆分到 N 路并行进程，共享输出目录。"""
import argparse, os, sys, subprocess, tempfile, time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from batch_transcribe import find_mp3_files


def write_file_list(files, idx, tmpdir):
    """写入一个临时文件列表"""
    path = os.path.join(tmpdir, f"batch_part_{idx}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for mp3 in files:
            f.write(mp3 + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="多路并行批量转录")
    parser.add_argument("--dir", required=True, help="包含 MP3 的目录")
    parser.add_argument("--workers", type=int, default=3, help="并行进程数（默认 3）")
    parser.add_argument("--speaker-profile", default=None)
    parser.add_argument("--speaker-threshold", type=float, default=0.65)
    parser.add_argument("--txt-dir", default=None)
    parser.add_argument("--json-dir", default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--merge-gap", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--output-json", action="store_true", default=True)
    args = parser.parse_args()

    # 1. 收集全部 MP3
    all_files = find_mp3_files(args.dir)
    if not all_files:
        print(f"未找到 MP3: {args.dir}")
        sys.exit(1)

    # 2. 过滤已处理的
    txt_dir = args.txt_dir or args.dir
    pending = []
    skipped = 0
    for mp3 in all_files:
        fname_base = os.path.splitext(os.path.basename(mp3))[0]
        txt_path = os.path.join(txt_dir, fname_base + "_dialogue.txt")
        if args.skip_existing and os.path.exists(txt_path):
            skipped += 1
            continue
        # 也检查音频同目录（兼容没有 --txt-dir 的情况）
        alt_txt = os.path.splitext(mp3)[0] + "_dialogue.txt"
        if os.path.exists(alt_txt) and alt_txt != txt_path:
            skipped += 1
            continue
        pending.append(mp3)

    print(f"总文件: {len(all_files)}, 已完成: {skipped}, 待处理: {len(pending)}")

    if not pending:
        print("全部已完成，无需处理。")
        sys.exit(0)

    # 3. 拆分
    n = min(args.workers, len(pending))
    chunk_size = len(pending) // n
    chunks = []
    for i in range(n):
        start = i * chunk_size
        end = start + chunk_size if i < n - 1 else len(pending)
        chunks.append(pending[start:end])

    print(f"拆分为 {n} 组: {[len(c) for c in chunks]}")

    # 4. 写文件列表
    tmpdir = tempfile.mkdtemp(prefix="batch_parallel_")
    list_paths = []
    for i, chunk in enumerate(chunks):
        list_paths.append(write_file_list(chunk, i, tmpdir))

    # 5. 构建公共参数
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_transcribe.py")
    base_cmd = [
        sys.executable, script,
        "--files", "",  # 占位，后面覆盖
        "--skip-existing",
        "--output-json",
    ]
    if args.speaker_profile:
        base_cmd += ["--speaker-profile", args.speaker_profile]
    if args.speaker_threshold:
        base_cmd += ["--speaker-threshold", str(args.speaker_threshold)]
    if args.txt_dir:
        base_cmd += ["--txt-dir", args.txt_dir]
    if args.json_dir:
        base_cmd += ["--json-dir", args.json_dir]
    if args.num_speakers:
        base_cmd += ["--num-speakers", str(args.num_speakers)]
    if args.merge_gap:
        base_cmd += ["--merge-gap", str(args.merge_gap)]

    # 6. 启动并行进程
    print(f"\n启动 {n} 个并行进程...")
    print("=" * 60)

    processes = []
    for i, list_path in enumerate(list_paths):
        cmd = base_cmd.copy()
        # 替换 --files 占位符
        cmd[base_cmd.index("--files") + 1] = list_path
        print(f"[Worker {i}] {len(chunks[i])} 个文件")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        processes.append((i, p, list_path))

    print("=" * 60)
    print("全部启动，等待完成...\n")

    # 7. 等待并收集输出
    ok_total, fail_total = 0, 0
    for i, p, list_path in processes:
        stdout, _ = p.communicate()
        # 打印每个 worker 的关键行
        for line in stdout.split("\n")[-10:]:
            if line.strip():
                print(f"  [W{i}] {line.strip()}")
        # 统计
        for line in stdout.split("\n"):
            if "成功:" in line or "完成:" in line:
                print(f"  [W{i}] {line.strip()}")
        if p.returncode == 0:
            ok_total += 1
        else:
            fail_total += 1

    print(f"\n{'=' * 60}")
    print(f"全部完成: {ok_total} 成功, {fail_total} 失败")

    # 8. 合并汇总
    summary_path = os.path.join(args.dir, "_batch_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"批量转写汇总（并行 {n} 路）\n")
        f.write(f"目录: {args.dir}\n")
        f.write(f"总文件: {len(all_files)}, 处理: {len(pending)}\n\n")
        for mp3 in sorted(pending):
            fname = os.path.basename(mp3)
            fname_base = os.path.splitext(fname)[0]
            t = os.path.join(txt_dir, fname_base + "_dialogue.txt")
            status = "✓" if os.path.exists(t) else "✗"
            f.write(f"  {status} {fname}\n")
    print(f"汇总: {summary_path}")


if __name__ == "__main__":
    main()
