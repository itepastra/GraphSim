#!/usr/bin/env python3
import re
import ast
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


START_RE = re.compile(r"^starting\s+(\d+)\s*$")
HIST_RE = re.compile(r"^histogram of (\d+) iters:\s*(\[.*\])\s*$")


def load_blocks(path):
    """
    Parse the file into blocks.

    Each block:
      starting {n}
      histogram of X iters: [...]
      histogram of Y iters: [...]
      ...

    Returns a list of dicts:
      { "start_n": int, "frames": [(iters, [(bin, count), ...]), ...] }
    """
    blocks = []
    current_block = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # New block marker
            m_start = START_RE.match(line)
            if m_start:
                # close previous block if it has any frames
                if current_block is not None and current_block["frames"]:
                    blocks.append(current_block)

                start_n = int(m_start.group(1))
                current_block = {"start_n": start_n, "frames": []}
                continue

            # Histogram line
            m_hist = HIST_RE.match(line)
            if m_hist:
                if current_block is None:
                    # In theory this shouldn't happen if each block starts with "starting n"
                    # but just in case, create a default block with start_n = -1
                    current_block = {"start_n": -1, "frames": []}

                iters = int(m_hist.group(1))
                pairs_str = m_hist.group(2)
                pairs = ast.literal_eval(pairs_str)
                current_block["frames"].append((iters, pairs))

    # Append last block if it has frames
    if current_block is not None and current_block["frames"]:
        blocks.append(current_block)

    if not blocks:
        raise ValueError("No histogram blocks found in file.")

    return blocks


def prepare_data(frames):
    """
    Convert list of (iters, [(bin, count), ...]) into:
      - xs: list of all bin indices (0..max_bin)
      - ys_all: list of y-value lists, one per frame
      - iters_list: list of iteration counts
      - max_count: global max count across frames
    """
    max_bin = max(bin_idx for _, pairs in frames for bin_idx, _ in pairs)
    max_count = max(count for _, pairs in frames for _, count in pairs)

    xs = list(range(max_bin + 1))
    ys_all = []
    iters_list = []
    tot_iters = 0

    for iters, pairs in frames:
        tot_iters += iters
        y = [0] * (max_bin + 1)
        for bin_idx, count in pairs:
            if 0 <= bin_idx <= max_bin:
                y[bin_idx] = count
        ys_all.append(y)
        iters_list.append(tot_iters)

    return xs, ys_all, iters_list, max_count


def make_animation_for_block(block, output_path, fps=10):
    """
    Make a GIF for a single block:
      block: { "start_n": int, "frames": [(iters, pairs), ...] }
    """
    frames = block["frames"]
    xs, ys_all, iters_list, max_count = prepare_data(frames)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(xs, ys_all[0])

    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_xlabel("Value")
    ax.set_ylabel("Times it appeared")

    title = ax.set_title(
        f"starting {block['start_n']} – histogram after {iters_list[0]} iterations"
    )

    def init():
        for bar, h in zip(bars, ys_all[0]):
            bar.set_height(h)
        title.set_text(
            f"starting {block['start_n']} – histogram after {iters_list[0]} iterations"
        )
        return bars

    def update(frame_idx):
        y = ys_all[frame_idx]
        for bar, h in zip(bars, y):
            bar.set_height(h)

        ymax = max(y) if y else 1
        ax.set_ylim(0, ymax*1.1)
        title.set_text(
            f"starting {block['start_n']} – histogram after {iters_list[frame_idx]} iterations"
        )
        return bars

    ani = FuncAnimation(
        fig,
        update,
        frames=len(ys_all),
        init_func=init,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    print(f"Saving GIF for block (starting {block['start_n']}) to {output_path} ...")
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)


def make_animations(input_path, output_prefix, fps=10):
    blocks = load_blocks(input_path)
    print(f"Found {len(blocks)} block(s).")

    for i, block in enumerate(blocks, start=1):
        start_n = block["start_n"]
        first_iters = block["frames"][0][0]
        last_iters = block["frames"][-1][0]

        # {n} from "starting {n}" is included in the file name:
        out_name = f"{output_prefix}_{start_n}.gif"
        out_path = os.path.abspath(out_name)

        make_animation_for_block(block, out_path, fps=fps)

    print("All blocks processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Create one animated GIF per 'starting n' histogram block."
    )
    parser.add_argument(
        "input",
        help="Input text file containing 'starting n' and 'histogram of X iters: [...]' lines",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default="histograms",
        help="Output filename prefix (default: histograms)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for each GIF (default: 10)",
    )

    args = parser.parse_args()
    make_animations(args.input, args.prefix, fps=args.fps)


if __name__ == "__main__":
    main()

