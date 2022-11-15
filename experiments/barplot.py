import os
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{xfrac}")

REDUCTION_COLORS = ["skyblue", "lightgreen", "bisque", "g", "r", "c", "m"]
REDUCTION_HATCHES = ["//", "..", "\\\\"]

SPEED_COLORS = ["skyblue", "violet", "pink"]
SPEED_HATCHES = ["//", "--", "**"]

CAP = ""


def plot_reduction_ratio(full_plot_data, folder):
    # Skip DD here. Because both DD and NNReduce can reach sub-graph minimality.
    # So just showing one is enough.
    plot_data = {}
    for k, v in full_plot_data.items():
        if "DD" in k or ("NNReduce" in k and len(k) > 8):
            continue
        plot_data[k] = v
    col_width = 0.85
    bar_width = col_width / len(plot_data)
    base_x = np.arange(len(files))
    legends = []

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(7.5, 1.8))

    once = False
    for idx, (label, data_dict) in enumerate(plot_data.items()):
        if not once:
            ax.bar(
                base_x,
                data_dict["osize[real]"],
                width=bar_width * len(plot_data),
                color="gray",
                label="Original Size",
                alpha=0.25,
            )
            legends.append("Original Size")
            once = True

        label.replace("NNReduce", r"\textsc{NNReduce}")

        legends.append(label)

        x_pos = base_x - 0.5 * col_width + (idx + 0.5) * bar_width
        ax.bar(
            x_pos,
            data_dict["msize[real]"],
            width=bar_width,
            label=label,
            alpha=0.6,
            color=REDUCTION_COLORS[idx],
            hatch=REDUCTION_HATCHES[idx],
            edgecolor="gray",
        )

    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlim(-1, len(files))
    ax.set_xticks(base_x)
    mark = r"\tilde"
    ax.set_xticklabels(
        [
            f"${mark if 'VERIFICATION' in files[i].upper() else ''} {{{i+1}}}$"
            for i in range(len(files))
        ],
        ha="center",
    )
    # set xtick size
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )  # labels along the bottom edge are off

    ax.legend(
        legends,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.33),
        ncol=4,
        fancybox=True,
        shadow=True,
    )
    ax.set_ylabel("Model Size (\# of Operators)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.grid(axis="x", linestyle="", alpha=0.6)

    # savefig with fig
    fig.savefig(os.path.join(folder, f"{tag}reduction.pdf"))
    fig.savefig(
        os.path.join(folder, f"{tag}reduction.png"), bbox_inches="tight", dpi=200
    )


def plot_speed(full_plot_data, folder):
    plot_data = {k: v for k, v in full_plot_data.items() if "Polygraph" not in k}
    assert "NNReduce" in plot_data, "NNReduce should be in the plot data"
    col_width = 0.75
    bar_width = col_width / len(plot_data)
    base_x = np.arange(len(files))
    legends = []

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 2.7))
    plt.subplots_adjust(hspace=-0.0)
    ax2.invert_yaxis()

    nnreduce_time = np.array(plot_data["NNReduce"]["time"])
    nnreduce_nexec = np.array(plot_data["NNReduce"]["nexec"])
    for k in plot_data:
        if "NNReduce" in k and "NNReduce" != k:
            nnreduce_time = np.minimum(nnreduce_time, plot_data[k]["time"])
            nnreduce_nexec = np.minimum(nnreduce_nexec, plot_data[k]["nexec"])

    tdd_time = np.array(plot_data["Topological DD"]["time"])
    tdd_nexec = np.array(plot_data["Topological DD"]["nexec"])

    nexec_fac = nnreduce_nexec / tdd_nexec
    time_fac = nnreduce_time / tdd_time

    ntop = -1
    ttop = -1
    for k, d in plot_data.items():
        ntop = max(ntop, np.max(d["nexec"]))
        ttop = max(ttop, np.max(d["time"]) * 1000)  # ms

    for idx, (ef, tf) in enumerate(zip(nexec_fac, time_fac)):
        # exec
        if abs(ef - 1) > 0.1:
            text = f"{ef:.1f}x" if ef > 1 else f"{ef:.1g}x"
            ax1.text(
                idx,
                max(*[v["nexec"][idx] for v in plot_data.values()]) * 0.97,
                text,
                color="green" if ef < 1 else "firebrick",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=7,
                rotation=45 if ef < 1 else -45,
            )
        # time
        if abs(tf - 1) > 0.1:
            text = f"{tf:.1f}x" if tf > 1 else f"{tf:.1g}x"
            degree = 35 if len(text) >= 6 else 45
            ax2.text(
                idx,
                max(*[v["time"][idx] for v in plot_data.values()])
                * 1000
                * (0.7 if len(text) == 6 else 0.85),
                text,
                color="green" if tf < 0.9 else "firebrick",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=7,
                rotation=-degree if tf < 0.9 else degree,
            )

    for idx, (label, data_dict) in enumerate(plot_data.items()):
        label.replace("NNReduce", r"\textsc{NNReduce}")

        legends.append(label)

        x_pos = base_x - 0.5 * col_width + (idx + 0.5) * bar_width

        ax1.bar(
            x_pos,
            data_dict["nexec"],
            width=bar_width,
            label=label,
            alpha=0.6,
            color=SPEED_COLORS[idx],
            hatch=SPEED_HATCHES[idx],
            edgecolor="gray",
        )

        ax2.bar(
            x_pos,
            1000 * np.array(data_dict["time"]),
            width=bar_width,
            label=label,
            alpha=0.6,
            color=SPEED_COLORS[idx],
            hatch=SPEED_HATCHES[idx],
            edgecolor="gray",
        )

    ax1.legend(
        legends,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.34),
        ncol=4,
        fancybox=True,
        shadow=True,
    )

    # reuse
    for ax in (ax1, ax2):
        ax.set_xlim(-0.6, len(files) - 0.4)
        # ax.tick_params(
        #     axis="x",  # changes apply to the x-axis
        #     which="both",  # both major and minor ticks are affected
        #     bottom=False,  # ticks along the bottom edge are off
        #     top=False,  # ticks along the top edge are off
        #     labelbottom=False,
        # )  # labels along the bottom edge are off
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

    ax2.set_xticks(base_x)
    mark = r"\tilde"
    ax2.set_xticklabels(
        [
            f"${mark if 'VERIFICATION' in files[i].upper() else ''}{{{i+1}}}$"
            for i in range(len(files))
        ],
        ha="center",
    )
    # set xtick size
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )  # labels along the bottom edge are off

    ax1.set_ylabel("\# Execution", fontweight="bold")
    ax2.set_ylabel("Time (Millisecond)", fontweight="bold")
    ax2.set_yscale("log")
    ax1.set_ylim((0, ntop * 1.2))
    ax2.set_ylim((pow(10, (math.log10(ttop) * 1.2)), 0))

    # savefig with fig
    fig.savefig(os.path.join(folder, f"{tag}speed.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(folder, f"{tag}speed.png"), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot barplot")
    parser.add_argument("--inps", nargs="+", type=str, help="list of csv files")
    parser.add_argument("--label", nargs="+", type=str, help="list of labels")
    parser.add_argument("--system", type=str, help="the system under test")
    parser.add_argument(
        "--semantic-only", action="store_true", help="only look at semantic bugs."
    )
    parser.add_argument(
        "--crash-only", action="store_true", help="only look at crash bugs."
    )
    parser.add_argument(
        "--keep-unmin", action="store_true", help="keep those unminimized."
    )
    parser.add_argument("--output", type=str, default="results", help="output folder.")
    args = parser.parse_args()

    if args.label is None or len(args.label) != len(args.inps):
        print("Bad labels... Using inp file names as labels.")
        args.label = args.inps

    tag = ""
    if args.system is not None:
        tag = f"{args.system}-"

    if "tvm" in tag.lower():
        CAP = "T"
    elif "ort" in tag.lower() or "onnx" in tag.lower():
        CAP = "O"

    # read data
    files = None
    data = []
    unmins = []
    for csv in args.inps:
        d = pd.read_csv(csv)
        if files is None:
            files = d["path"].tolist()
            if args.semantic_only:
                print("Only checking semantic bugs.")
                files = [f for f in files if "INCONSISTENCY" in f]
            elif args.crash_only:
                print("Only checking crash bugs.")
                files = [f for f in files if "INCONSISTENCY" not in f]
        else:
            tar = set(d["path"].tolist())
            cur = set(files)
            if tar != cur:
                print(f"Files are not the same: {cur - tar} vs {tar - cur} for {csv}")
                print("Use intersection...")
                files = [f for f in files if f in (cur & tar)]
            unmin_cur = []
            for _, row in d.iterrows():
                if row["before[real]"] == row["after[real]"]:
                    unmin_cur.append(row["path"])
            unmins.append(unmin_cur)

        data.append(d.to_numpy())

    if not args.keep_unmin:
        unmins = set.intersection(*[set(u) for u in unmins])
        if len(unmins) > 0:
            print(f"Skipping unminimized {len(unmins)} files")
            print("========== [Bugs Skipped] ==========")
            for unmin in unmins:
                print(f"\t{unmin}")

        files = [f for f in files if f not in unmins]

    print("========== [Bugs Used] ==========")
    for i, f in enumerate(files):
        print("\t", f"{CAP}{i+1}", "\t", f)

    # make dir
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    plot_data = {}
    for label, df in zip(args.label, data):
        v2plot = {
            "osize[full]": [],
            "msize[full]": [],
            "osize[real]": [],
            "msize[real]": [],
            "time": [],
            "nexec": [],
        }
        for f in files:
            row = df[np.where(df[:, 0] == f)][0]
            v2plot["osize[full]"].append(row[1])
            v2plot["msize[full]"].append(row[2])
            v2plot["osize[real]"].append(row[3])
            v2plot["msize[real]"].append(row[4])
            v2plot["nexec"].append(row[5])
            v2plot["time"].append(row[6])

        plot_data[label] = v2plot

        # Analytics:
        print(f"======== {label} ========")
        msize = np.array(v2plot["msize[real]"])
        osize = np.array(v2plot["osize[real]"])
        times = np.array(v2plot["time"])
        print(f"{times.sum() = :.2f}s")

        ratios = msize / osize
        ratios.sort()

        print(f"node / sec: {(osize - msize).sum() / times.sum():.3f}")
        print(f"model / sec: {len(msize) / times.sum() :.3f}")
        print(f"avg.   ratio: {ratios.mean() :.3f}")
        print(f"medium ratio: {ratios[len(ratios) // 2] :.3f}")
        print(f"min.   ratio: {ratios.min():.3f}")
        print(f"max.   ratio: {ratios.max():.3f}")

    plot_reduction_ratio(plot_data, args.output)
    plot_speed(plot_data, args.output)
