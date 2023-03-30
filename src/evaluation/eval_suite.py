import os
import skimage.measure
import lpips
import numpy as np
import torch
import imageio
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tqdm

METRIC_OPT_DICT = dict(l1="-", l2="-", lpips="-", psnr="+", ssim="+")
METRIC_LIMIT_DICT = dict(l1=[0, 0.1], l2=[0, 0.05], lpips=[0., 0.5], psnr=[12, 30], ssim=[.6, 1.])
AVERAGE_SCORE_FILENAME = "average_scores.json"
REPORT_DETAIL_FILENAME = "detailed_report.json"
BARPLOT_FILENAME = "average_scores.png"
EXAMPLE_PLOT_FILENAME = "examples.png"
N_EXAMPLE_PLOTS = 5
PRED_SUFFIX = "-pred.png"
GT_SUFFIX = "-gt.png"
REF_SUFFIX = "-ref.png"
DEPTH_SUFFIX = "-depth.png"


def scatter2rangeplot(x, y, n_bins=5):
    x = np.array(x)
    y = np.array(y)
    xmin, xmax = np.min(x), np.max(x)
    bin_edges = np.linspace(xmin, xmax, n_bins + 1)
    x_means, y_means, stds = list(), list(), list()
    for i in range(n_bins):
        mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        x_means.append(np.mean(x[mask]))
        y_means.append(np.mean(y[mask]))
        stds.append(np.std(y[mask]))

    return np.array(x_means), np.array(y_means), np.array(stds)


@torch.no_grad()
def evaluate_folder(source_dir, outdir, device=None, pred_suffix=PRED_SUFFIX, gt_suffix=GT_SUFFIX,
                    ref_suffix=REF_SUFFIX, depth_suffix=DEPTH_SUFFIX,
                    show_tqdm=False):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    outdir = Path(outdir)
    os.makedirs(outdir, exist_ok=True)

    lpips_vgg = lpips.LPIPS(net="vgg").to(device=device)
    source_dir = Path(source_dir)

    gt_img_paths = [p for p in sorted(source_dir.iterdir()) if p.name.endswith(gt_suffix)]
    pred_img_paths = [p.parent / p.name.replace(gt_suffix, pred_suffix) for p in gt_img_paths]

    scores = defaultdict(list)
    iterator = zip(gt_img_paths, pred_img_paths)
    if show_tqdm:
        iterator = tqdm.tqdm(iterator, total=len(gt_img_paths), mininterval=30.)
    for gt_path, pred_path in iterator:
        gt = imageio.imread(gt_path).astype(np.float32)[..., :3] / 255.0
        pred = imageio.imread(pred_path).astype(np.float32) / 255.0
        ssim = skimage.metrics.structural_similarity(pred, gt, channel_axis=-1, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(pred, gt, data_range=1)
        mse = skimage.metrics.mean_squared_error(pred, gt)
        manhattan = np.mean(np.abs((pred - gt)))

        scores["ssim"].append(ssim)
        scores["psnr"].append(psnr)
        scores["l2"].append(mse)
        scores["l1"].append(manhattan)

        gt = torch.from_numpy(gt).permute(2, 0, 1)[None] * 2.0 - 1.0
        pred = torch.from_numpy(pred).permute(2, 0, 1)[None] * 2.0 - 1.0
        scores["lpips"].append(lpips_vgg(pred.to(device=device), gt.to(device=device)).flatten().cpu().item())

    # normalizing metrics
    avg_metrics = dict()
    for key, val in scores.items():
        avg_metrics[key] = np.mean(val).astype(np.float64)

    # writing average metrics
    summary_fp = outdir / AVERAGE_SCORE_FILENAME
    with open(summary_fp, "w") as f:
        json.dump(avg_metrics, f, indent="\t")

    # writing detailed metrics report
    detail_fp = outdir / REPORT_DETAIL_FILENAME
    detailed_report = list()
    for i in range(len(pred_img_paths)):
        detailed_report.append(dict(path=str(pred_img_paths[i])
                                    ))
        for key, val in scores.items():
            detailed_report[-1][key] = float(val[i])

    with open(detail_fp, "w") as f:
        json.dump(detailed_report, f, indent="\t")

    # plotting  examples
    example_plot_fp = outdir / EXAMPLE_PLOT_FILENAME
    nexamples = N_EXAMPLE_PLOTS
    idcs = np.linspace(0, len(gt_img_paths) - 1, nexamples).astype(int)
    img_rows = []
    for i in range(nexamples):
        idx = idcs[i]
        pred_img_path = pred_img_paths[idx]
        ref_img_path = pred_img_path.parent / pred_img_path.name.replace(pred_suffix, ref_suffix)
        gt_img_path = pred_img_path.parent / pred_img_path.name.replace(pred_suffix, gt_suffix)
        depth_img_path = pred_img_path.parent / pred_img_path.name.replace(pred_suffix, depth_suffix)

        pred_img = imageio.imread(pred_img_path)
        ref_img = imageio.imread(ref_img_path) if ref_img_path.exists() else np.zeros_like(pred_img)
        gt_img = imageio.imread(gt_img_path) if gt_img_path.exists() else np.zeros_like(pred_img)
        depth_img = imageio.imread(depth_img_path) if depth_img_path.exists() else np.zeros_like(pred_img)
        H, W, _ = pred_img.shape
        nref = ref_img.shape[1] // W

        img_rows.append(np.concatenate([*np.hsplit(ref_img, nref), gt_img, pred_img, depth_img], axis=1))
    img = np.concatenate(img_rows, axis=0)
    imageio.imwrite(example_plot_fp, img)

    return avg_metrics


def compare_evaluations(eval_dirs, outdir):
    """
    compares evaluations of several models and stores results in outdir
    Args:
        eval_dirs: list of tuples with entries: (model_name, path to evaluation dir resulting from evaluate_folder)
        outdir:

    Returns:

    """
    outdir = Path(outdir)
    os.makedirs(outdir, exist_ok=True)

    # ----------------------  Create Violin comparison plot ---------------------------
    score_dict = defaultdict(dict)
    model_names = list()
    for model_name, d in eval_dirs:
        model_names.append(model_name)
        with open(os.path.join(d, REPORT_DETAIL_FILENAME), "r") as f:
            report = json.load(f)
            metrics = sorted([k for k in report[0] if k in METRIC_OPT_DICT])
            for metric in metrics:
                scores = list()
                for sample in report:
                    scores.append(sample[metric])
                score_dict[metric][model_name] = scores

    nrows = len(metrics)
    ncols = 1
    nmodels = len(model_names)
    fig = plt.figure(figsize=(3 * nmodels, 15))
    DEFAULT_BAR_COLOR = u'#1f77b4'
    BEST_BAR_COLOR = 'green'
    WORST_BAR_COLOR = 'red'
    x_pad = .5
    for i, metric in enumerate(metrics):
        x = np.arange(nmodels)
        # drawing best performing model in orange instead of blue
        metric_opt = METRIC_OPT_DICT[metric]
        model_scores = [score_dict[metric][model] for model in model_names]
        model_scores_mean = np.array([np.mean(scores) for scores in model_scores])
        model_scores_std = np.array([np.std(scores) for scores in model_scores])
        nsamples = np.array([len(s) for s in model_scores])
        best_model_idx = np.argmax(model_scores_mean) if metric_opt == "+" else np.argmin(model_scores_mean)
        worst_model_idx = np.argmin(model_scores_mean) if metric_opt == "+" else np.argmax(model_scores_mean)
        colors = [DEFAULT_BAR_COLOR] * len(model_names)
        colors[best_model_idx] = BEST_BAR_COLOR
        colors[worst_model_idx] = WORST_BAR_COLOR

        ax = plt.subplot(nrows, ncols, i + 1)
        parts = ax.violinplot(model_scores, positions=x, showextrema=False, widths=.9
                              )
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[j])
        ax.scatter(x, model_scores_mean, c="black")
        ax.vlines(x, model_scores_mean - model_scores_std, model_scores_mean + model_scores_std, color="black")
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names if i == len(metrics) - 1 else [""] * len(model_names))
        ax.tick_params(labelrotation=45)
        ax.set_ylabel(metric.upper() + f" ({metric_opt})")
        # ax.set_ylim(METRIC_LIMIT_DICT[metric])
        ax.set_xlim((-x_pad, len(model_names) - 1 + x_pad))

        # adding bar value texts
        for j, (mean, std, n) in enumerate(zip(model_scores_mean, model_scores_std, nsamples)):
            ax.text(j, mean, f"  {mean:.3f}+-{std / np.sqrt(n):.3f}",
                    horizontalalignment="left",
                    verticalalignment="center")
    plt.tight_layout()
    outfile = outdir / BARPLOT_FILENAME
    plt.savefig(outfile)
    plt.close(fig)

    # # ----------------------  Create average score comparison bar plot ---------------------------
    # model_names = list()
    # metrics = list()
    # avg_scores = defaultdict(dict)
    # for model_name, eval_dir in eval_dirs:
    #     model_names.append(model_name)
    #     report_path = os.path.join(eval_dir, AVERAGE_SCORE_FILENAME)
    #     with open(report_path, "r") as f:
    #         report = json.load(f)
    #     metrics = sorted([k for k in report if k in METRIC_OPT_DICT])
    #     avg_scores[model_name] = report
    #
    # nrows = len(metrics)
    # ncols = 1
    # fig = plt.figure(figsize=(10, 15))
    # DEFAULT_BAR_COLOR = u'#1f77b4'
    # BEST_BAR_COLOR = 'green'
    # WORST_BAR_COLOR = 'red'
    # x_pad = .5
    # for i, metric in enumerate(metrics):
    #     # drawing best performing model in orange instead of blue
    #     metric_opt = METRIC_OPT_DICT[metric]
    #     model_scores = np.array([avg_scores[model][metric] for model in model_names])
    #     best_model_idx = np.argmax(model_scores) if metric_opt == "+" else np.argmin(model_scores)
    #     worst_model_idx = np.argmin(model_scores) if metric_opt == "+" else np.argmax(model_scores)
    #     colors = [DEFAULT_BAR_COLOR] * len(model_names)
    #     colors[best_model_idx] = BEST_BAR_COLOR
    #     colors[worst_model_idx] = WORST_BAR_COLOR
    #
    #     ax = plt.subplot(nrows, ncols, i + 1)
    #     ax.bar(x=np.arange(len(model_names)), height=model_scores, color=colors, align="center",
    #            tick_label=(model_names if i == len(metrics) - 1 else [""] * len(model_names)))
    #     ax.tick_params(labelrotation=45)
    #     ax.set_ylabel(metric.upper() + f" ({metric_opt})")
    #     ax.set_ylim(METRIC_LIMIT_DICT[metric])
    #     ax.set_xlim((-x_pad, len(model_names) - 1 + x_pad))
    #
    #     # adding bar value texts
    #     for j, score in enumerate(model_scores):
    #         ax.text((j + x_pad) / len(model_names), .05, f"{score:.3f}", horizontalalignment="center",
    #                 verticalalignment="bottom", transform=ax.transAxes)
    # plt.tight_layout()
    # outfile = outdir / BARPLOT_FILENAME
    # plt.savefig(outfile)
    # plt.close(fig)

    # ---------------- COMPARING EXAMPLE PLOTS ---------------------------
    try:
        model_names = list()
        example_plot_dict = dict()
        for model_name, eval_dir in eval_dirs:
            model_names.append(model_name)
            example_plot_path = os.path.join(eval_dir, EXAMPLE_PLOT_FILENAME)
            example_plot = imageio.imread(example_plot_path)
            split_example_plot = np.vsplit(example_plot, N_EXAMPLE_PLOTS)
            example_plot_dict[model_name] = split_example_plot
        for i in range(N_EXAMPLE_PLOTS):
            canvas = np.concatenate([example_plot_dict[model_name][i] for model_name in model_names], axis=0)
            # writing model names onto canvas
            H, W = canvas.shape[:2]
            h = H / len(model_names)
            text_canvas = Image.new('RGB', [H, 40], (255, 255, 255))
            draw = ImageDraw.Draw(text_canvas)
            font = ImageFont.truetype("DejaVuSerif.ttf", size=15)
            text_centers = np.linspace(h / 2, H - h / 2, len(model_names))[::-1]
            for center, model_name in zip(text_centers, model_names):
                w, h = font.getsize(model_name)
                offset = (center - w / 2, 0)
                black = "#000000"
                draw.text(offset, model_name, font=font, fill=black)
            text_canvas = np.rot90(np.array(text_canvas))
            canvas = np.concatenate((text_canvas, canvas), axis=1)
            outfilename = ".".join(EXAMPLE_PLOT_FILENAME.split(".")[:-1]) + f"_{i}." + EXAMPLE_PLOT_FILENAME.split(".")[-1]
            outfile = outdir / outfilename
            imageio.imwrite(outfile, canvas)
    except Exception:
        pass

    # ------------- What is the difference plots ----------------------------
    n_samples = 3

    detail_reports = []
    model_names = []
    for model_name, d in eval_dirs:
        model_names.append(model_name)
        with open(os.path.join(d, REPORT_DETAIL_FILENAME), "r") as f:
            detail_reports.append(json.load(f))

    metrics = sorted([k for k in detail_reports[0][0] if k in METRIC_OPT_DICT])

    # check if reports evaluate same images
    assert all([len(report) == len(detail_reports[0]) for report in detail_reports[1:]])
    samples_names = [[Path(sample["path"]).name for sample in report] for report in detail_reports]
    assert all([s == samples_names[0] for s in samples_names[1:]])
    samples_names = np.array(samples_names[0])  # Nsamples,

    scores = dict()  # {metric_name: {model_name: [score_sample1, score_sample2]}}
    for j in range(len(metrics)):
        metric_score_dict = dict()
        for k in range(len(model_names)):
            model_scores = list()
            for i in range(len(detail_reports[0])):
                model_scores.append(detail_reports[k][i][metrics[j]])
            model_scores = np.array(model_scores)
            metric_score_dict[model_names[k]] = model_scores
        scores[metrics[j]] = metric_score_dict

    # finding sample with biggest difference
    highest_std_sample_names = dict()
    highest_std_idcs = dict()
    for m in metrics:
        stds = np.std(np.stack([scores[m][model_name] for model_name in model_names], axis=-1), axis=-1)
        highest_std_idcs[m] = np.argsort(stds)[::-1][:n_samples]
        highest_std_sample_names[m] = samples_names[highest_std_idcs[m]]

    # plot difference images
    ncols = len(model_names) + 1
    for m in metrics:
        for j in range(n_samples):
            fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(ncols * 5, 5 * 3))
            l1_error_maps = []
            vis_path = Path(eval_dirs[0][1]) / "visualizations"
            gt_path = vis_path / highest_std_sample_names[m][j].replace(PRED_SUFFIX, GT_SUFFIX)
            ref_path = vis_path / highest_std_sample_names[m][j].replace(PRED_SUFFIX, REF_SUFFIX)
            gt = np.array(Image.open(gt_path).convert("RGB")) / 255
            ref = np.array(Image.open(ref_path).convert("RGB")) / 255
            axes[0, -1].imshow(gt)
            axes[2, -1].imshow(ref)

            # plotting predictions and l1 error maps

            for i in range(len(model_names)):
                pred_path = Path(eval_dirs[i][1]) / "visualizations" / highest_std_sample_names[m][j]
                pred = np.array(Image.open(pred_path).convert("RGB")) / 255
                sample_idx = highest_std_idcs[m][j]
                score = scores[m][model_names[i]][sample_idx]

                # check if best model
                is_best_model = False
                if METRIC_OPT_DICT[m] == "+":
                    if np.all(score >= np.array([scores[m][model_name][sample_idx] for model_name in model_names])):
                        is_best_model = True
                else:
                    if np.all(score <= np.array([scores[m][model_name][sample_idx] for model_name in model_names])):
                        is_best_model = True

                l1_error_map = np.abs(pred - gt).mean(axis=-1)
                l1_error_maps.append(l1_error_map)
                axes[0, i].imshow(pred)
                axes[2, i].imshow(pred)
                axes[0, i].set_title(model_names[i] + "\n" + f"{m}[{METRIC_OPT_DICT[m]}]: {score:.5e}", fontsize=9,
                                     color="green" if is_best_model else "black")

            # plotting l1 error maps
            l1_error_maps = np.stack(l1_error_maps, axis=0)  # N, H, W
            l1_error_maps_max = np.max(l1_error_maps, axis=0, keepdims=True)  # 1, H, W
            l1_error_maps_min = np.min(l1_error_maps, axis=0, keepdims=True)  # 1, H, W
            l1_error_min = np.min(l1_error_maps_min)
            l1_error_max = np.max(l1_error_maps_max)
            l1_error_maps_normalized = (l1_error_maps - l1_error_maps_min) / (l1_error_maps_max - l1_error_maps_min)

            cdict = {'red': [[0.0, 0.0, 0.0],
                             [0.5, 1.0, 1.0],
                             [1.0, 1.0, 1.0]],
                     'green': [[0.0, 1.0, 1.0],
                               [0.5, 1.0, 1.0],
                               [1.0, 0.0, 0.0]],
                     'blue': [[0.0, 0.0, 0.0],
                              [0.5, 0.0, 0.0],
                              [1.0, 0.0, 0.0]]}
            for i in range(len(model_names)):
                from matplotlib.colors import LinearSegmentedColormap
                axes[2, i].imshow(l1_error_maps_normalized[i], alpha=1.,
                                  cmap=LinearSegmentedColormap("testCMap", segmentdata=cdict, N=256))
                heatmap = axes[1, i].imshow(l1_error_maps[i], vmin=l1_error_min, vmax=l1_error_max)
            axes[1, 0].set_xlabel("L1 error map")
            axes[2, 0].set_xlabel("l1 comparison map")
            plt.sca(axes[1, -1])
            plt.colorbar(heatmap)

            [a.axis("off") for a in axes.flatten()]
            fig.suptitle(f"{m} sample {j + 1}/{n_samples} {highest_std_sample_names[m][j]}")
            # plt.show()
            outpath = outdir / f"biggest_difference_{m}_{j + 1}.png"
            plt.savefig(outpath)
            plt.close()
