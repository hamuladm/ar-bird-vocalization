from collections import Counter
from datasets import load_dataset, load_dataset_builder
import numpy as np
import matplotlib.pyplot as plt


def analyze_xcm():
    ds = load_dataset("DBD-research-group/BirdSet", "XCM", split="train", trust_remote_code=True)
    
    builder = load_dataset_builder("DBD-research-group/BirdSet", "XCM", trust_remote_code=True)
    class_names = builder.info.features['ebird_code'].names
    
    num_samples = len(ds)
    
    events_per_sample = []
    event_durations = []
    class_counter = Counter()
    
    for item in ds:
        detected = item.get("detected_events") or []
        if detected:
            n_events = len(detected)
            for event in detected:
                duration = event[1] - event[0]
                event_durations.append(duration)
        else:
            n_events = 1
            start = item.get("start_time", 0) or 0
            end = item.get("end_time") or 5
            event_durations.append(end - start)
        events_per_sample.append(n_events)
        class_counter[item["ebird_code"]] += 1
    
    total_events = sum(events_per_sample)
    avg_events = np.mean(events_per_sample)
    avg_event_duration = np.mean(event_durations)
    median_event_duration = np.median(event_durations)
    min_event_duration = np.min(event_durations)
    max_event_duration = np.max(event_durations)
    
    print("\n" + "=" * 50)
    print("XCM DATASET STATISTICS")
    print("=" * 50)
    print(f"Total audio samples:      {num_samples:,}")
    print(f"Total events:             {total_events:,}")
    print(f"Avg events per sample:    {avg_events:.2f}")
    print(f"Number of classes:        {len(class_names)}")
    print(f"\nEvent Duration (seconds):")
    print(f"  Average:                {avg_event_duration:.2f}s")
    print(f"  Median:                 {median_event_duration:.2f}s")
    print(f"  Min / Max:              {min_event_duration:.2f}s / {max_event_duration:.2f}s")
    
    print("\n" + "-" * 50)
    print("CLASS DISTRIBUTION (Top 20)")
    print("-" * 50)
    
    sorted_classes = class_counter.most_common()
    for class_idx, count in sorted_classes:
        name = class_names[class_idx]
        pct = count / num_samples * 100
        print(f"  {name:12s}: {count:5d} ({pct:5.2f}%)")
    
    counts = list(class_counter.values())
    print("\n" + "-" * 50)
    print("DISTRIBUTION STATS")
    print("-" * 50)
    print(f"Min samples per class:    {min(counts)}")
    print(f"Max samples per class:    {max(counts)}")
    print(f"Median samples per class: {np.median(counts):.0f}")
    print(f"Std dev:                  {np.std(counts):.1f}")
    
    classes_200plus = [c for c in counts if c >= 200]
    samples_200plus = sum(classes_200plus)
    print("\n" + "-" * 50)
    print("200+ SAMPLES THRESHOLD")
    print("-" * 50)
    print(f"Classes with 200+ samples: {len(classes_200plus)} / {len(counts)}")
    print(f"Total samples (200+ only): {samples_200plus:,} / {num_samples:,} ({samples_200plus/num_samples*100:.1f}%)")
    
    sorted_counts = sorted(counts, reverse=True)
    cumsum = np.cumsum(sorted_counts)
    total = sum(counts)
    
    n = len(counts)
    sorted_asc = sorted(counts)
    gini = (2 * sum((i + 1) * c for i, c in enumerate(sorted_asc)) - (n + 1) * total) / (n * total)
    
    probs = np.array(counts) / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    effective_classes = np.exp(entropy)
    
    top_10_pct = int(n * 0.1)
    top_20_pct = int(n * 0.2)
    top_50_pct = int(n * 0.5)
    
    print("\n" + "-" * 50)
    print("CONCENTRATION METRICS")
    print("-" * 50)
    print(f"Gini coefficient:         {gini:.3f} (0=equal, 1=concentrated)")
    print(f"Effective # of classes:   {effective_classes:.0f} / {n} ({effective_classes/n*100:.1f}%)")
    print(f"\nPareto Analysis:")
    print(f"  Top 10% classes ({top_10_pct}):  {cumsum[top_10_pct-1]/total*100:.1f}% of samples")
    print(f"  Top 20% classes ({top_20_pct}):  {cumsum[top_20_pct-1]/total*100:.1f}% of samples")
    print(f"  Top 50% classes ({top_50_pct}): {cumsum[top_50_pct-1]/total*100:.1f}% of samples")
    print("=" * 50)
    
    plot_distribution(class_counter, class_names)


def plot_distribution(class_counter: Counter, class_names: list):
    counts = [class_counter.get(i, 0) for i in range(len(class_names))]
    
    sorted_counts = sorted(counts, reverse=True)
    cumsum = np.cumsum(sorted_counts) / sum(counts) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.bar(range(len(counts)), counts, color='steelblue', width=1.0)
    ax1.axhline(y=200, color='red', linestyle='--', linewidth=1.5, label='200 samples')
    ax1.set_xlabel("Class Index (0-410)", fontsize=11)
    ax1.set_ylabel("Number of Audio Samples", fontsize=11)
    ax1.set_title("Audio Samples per Class", fontsize=12)
    ax1.set_xlim(0, len(counts))
    ax1.legend()
    
    ax2 = axes[1]
    x_pct = np.arange(1, len(cumsum) + 1) / len(cumsum) * 100
    ax2.plot(x_pct, cumsum, color='steelblue', linewidth=2)
    ax2.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=20, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.fill_between(x_pct, cumsum, alpha=0.3)
    ax2.set_xlabel("% of Classes (sorted by size)", fontsize=11)
    ax2.set_ylabel("Cumulative % of Samples", fontsize=11)
    ax2.set_title("Pareto Curve (Cumulative Contribution)", fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    idx_80 = np.argmax(cumsum >= 80)
    pct_classes_for_80 = (idx_80 + 1) / len(cumsum) * 100
    ax2.annotate(f'{pct_classes_for_80:.0f}% of classes\n= 80% of data', 
                xy=(pct_classes_for_80, 80), xytext=(pct_classes_for_80 + 15, 65),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig("data/xcm_class_distribution.png", dpi=150)


if __name__ == "__main__":
    analyze_xcm(show_plot=True)
