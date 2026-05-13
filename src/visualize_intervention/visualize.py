import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt


COLOR_LABELS = ["RED", "GREEN", "BLUE"]


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    if text is None:
        return "EMPTY_OUTPUT"

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return "EMPTY_OUTPUT"

    return text


def shorten_label(text, max_len=45):
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def extract_model_prediction(output_text, options=None):
    raw = clean_text(output_text)
    upper = raw.upper()

    for color in COLOR_LABELS:
        if re.search(rf"\b{color}\b", upper):
            return color

    if options:
        letter_match = re.search(r"\b([A-D])\b", upper)
        if letter_match:
            idx = ord(letter_match.group(1)) - ord("A")
            if 0 <= idx < len(options):
                return clean_text(options[idx]).upper()

    if options:
        number_match = re.search(r"\b([1-4])\b", upper)
        if number_match:
            idx = int(number_match.group(1)) - 1
            if 0 <= idx < len(options):
                return clean_text(options[idx]).upper()

    if options:
        for option in options:
            option_clean = clean_text(option)
            if re.search(rf"\b{re.escape(option_clean)}\b", raw, flags=re.IGNORECASE):
                return option_clean.upper()

    return raw


def normalize_answer(text):
    text = clean_text(text).upper()

    for color in COLOR_LABELS:
        if text == color:
            return color

    return text


def build_dataframe(results, layers):
    rows = []

    for i, item in enumerate(results):
        intervened_output = item.get("intervened_output", "")
        correct_answer = item.get("correct_answer", "")
        chosen_concept = item.get("chosen_concept", "")

        predicted = extract_model_prediction(
            output_text=intervened_output,
            options=item.get("options", []),
        )

        target = normalize_answer(correct_answer)

        rows.append(
            {
                "index": i,
                "question": item.get("question", ""),
                "options": item.get("options", ""),
                "chosen_concept": str(chosen_concept).upper(),
                "correct_answer": target,
                "predicted_answer": predicted,
                "predicted_answer_display": shorten_label(predicted),
                "intervened_output": intervened_output,
                "is_correct": predicted.upper() == target.upper(),
                "alpha": item.get("alpha", None),
                "layers": ",".join(map(str, layers)),
            }
        )

    return pd.DataFrame(rows)


def save_plot(fig, output_dir, name):
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)


def plot_overall_accuracy(df, output_dir, title_suffix):
    total = len(df)
    correct = int(df["is_correct"].sum())
    incorrect = total - correct

    accuracy = correct / total if total else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["Correct", "Incorrect"]
    values = [correct, incorrect]

    ax.bar(labels, values)
    ax.set_ylabel("Number of Questions")
    ax.set_title(f"Overall Intervention Accuracy\n{title_suffix}\nAccuracy = {accuracy:.3f}")

    for i, value in enumerate(values):
        ax.text(i, value, str(value), ha="center", va="bottom")

    save_plot(fig, output_dir, "overall_accuracy")


def plot_prediction_distribution(df, output_dir, title_suffix):
    counts = df["predicted_answer_display"].value_counts()

    labels = list(counts.index)
    values = list(counts.values)

    height = max(4, 0.45 * len(labels))

    fig, ax = plt.subplots(figsize=(9, height))

    ax.barh(labels, values)
    ax.set_xlabel("Number of Predictions")
    ax.set_ylabel("Model Prediction")
    ax.set_title(f"Predicted Answer Distribution\n{title_suffix}")

    for i, value in enumerate(values):
        ax.text(value, i, str(value), va="center", ha="left")

    ax.invert_yaxis()

    save_plot(fig, output_dir, "prediction_distribution")


def plot_accuracy_by_correct_answer(df, output_dir, title_suffix):
    labels = COLOR_LABELS
    accuracies = []
    counts = []

    for label in labels:
        subset = df[df["correct_answer"] == label]
        count = len(subset)
        acc = subset["is_correct"].mean() if count else 0.0

        accuracies.append(acc)
        counts.append(count)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(labels, accuracies)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by Correct Answer\n{title_suffix}")

    for i, acc in enumerate(accuracies):
        ax.text(
            i,
            acc,
            f"{acc:.2f}\n(n={counts[i]})",
            ha="center",
            va="bottom",
        )

    save_plot(fig, output_dir, "accuracy_by_correct_answer")


def plot_confusion_matrix(df, output_dir, title_suffix):
    true_labels = sorted(df["correct_answer"].dropna().unique())
    pred_labels = sorted(df["predicted_answer_display"].dropna().unique())

    matrix = []

    for true_label in true_labels:
        row = []
        true_subset = df[df["correct_answer"] == true_label]

        for pred_label in pred_labels:
            row.append(int((true_subset["predicted_answer_display"] == pred_label).sum()))

        matrix.append(row)

    width = max(7, 0.6 * len(pred_labels))
    height = max(5, 0.6 * len(true_labels))

    fig, ax = plt.subplots(figsize=(width, height))

    im = ax.imshow(matrix)

    ax.set_xticks(range(len(pred_labels)))
    ax.set_yticks(range(len(true_labels)))

    ax.set_xticklabels(pred_labels, rotation=45, ha="right")
    ax.set_yticklabels(true_labels)

    ax.set_xlabel("Model Prediction")
    ax.set_ylabel("Correct Answer")
    ax.set_title(f"Confusion Matrix\n{title_suffix}")

    for i in range(len(true_labels)):
        for j in range(len(pred_labels)):
            ax.text(j, i, matrix[i][j], ha="center", va="center")

    fig.colorbar(im, ax=ax)

    save_plot(fig, output_dir, "confusion_matrix")


def save_metrics(df, output_dir, layers):
    total = len(df)
    correct = int(df["is_correct"].sum())
    incorrect = total - correct
    empty_outputs = int((df["predicted_answer"] == "EMPTY_OUTPUT").sum())

    accuracy = correct / total if total else 0.0

    summary_rows = [
        {
            "metric": "total_questions",
            "value": total,
        },
        {
            "metric": "correct_predictions",
            "value": correct,
        },
        {
            "metric": "incorrect_predictions",
            "value": incorrect,
        },
        {
            "metric": "empty_outputs",
            "value": empty_outputs,
        },
        {
            "metric": "overall_accuracy",
            "value": accuracy,
        },
        {
            "metric": "intervened_layers",
            "value": ",".join(map(str, layers)),
        },
    ]

    for color in COLOR_LABELS:
        subset = df[df["correct_answer"] == color]
        color_acc = subset["is_correct"].mean() if len(subset) else 0.0

        summary_rows.append(
            {
                "metric": f"accuracy_for_{color}",
                "value": color_acc,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        os.path.join(output_dir, "metrics_summary.csv"),
        index=False,
    )

    df.to_csv(
        os.path.join(output_dir, "normalized_results.csv"),
        index=False,
    )


def compare_with_baseline(intervened_df, baseline_path, output_dir, layers):
    baseline_results = load_results(baseline_path)
    baseline_df = build_dataframe(baseline_results, layers=[])

    baseline_acc = baseline_df["is_correct"].mean() if len(baseline_df) else 0.0
    intervened_acc = intervened_df["is_correct"].mean() if len(intervened_df) else 0.0

    labels = ["Baseline", "Intervened"]
    values = [baseline_acc, intervened_acc]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Baseline vs Intervention Accuracy\nLayers: {', '.join(map(str, layers))}")

    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")

    save_plot(fig, output_dir, "baseline_vs_intervention_accuracy")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize intervention results for RED/GREEN/BLUE concept experiments."
    )

    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Path to the intervened output JSON file.",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layers that were intervened on, e.g. --layers 10 30 59.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="intervention_plots",
        help="Directory where plots and CSV files will be saved.",
    )

    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="Optional path to non-intervened baseline output JSON.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_path)
    df = build_dataframe(results, args.layers)

    title_suffix = f"Layers: {', '.join(map(str, args.layers))}"

    save_metrics(df, args.output_dir, args.layers)

    plot_overall_accuracy(df, args.output_dir, title_suffix)
    plot_prediction_distribution(df, args.output_dir, title_suffix)
    plot_accuracy_by_correct_answer(df, args.output_dir, title_suffix)
    plot_confusion_matrix(df, args.output_dir, title_suffix)

    if args.baseline_path is not None:
        compare_with_baseline(
            intervened_df=df,
            baseline_path=args.baseline_path,
            output_dir=args.output_dir,
            layers=args.layers,
        )

    print(f"Saved plots and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()