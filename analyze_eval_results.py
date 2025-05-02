import os
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Needed for checking unique counts

def extract_metric_from_json(filepath, metric_key):
    """Loads the entire JSON file and extracts the metric value."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if metric_key in data:
                return float(data[metric_key])
            else:
                # Check if the key exists without the '@' (e.g., map5 instead of map@5)
                # Also handle recall@k -> recallk
                fallback_key = metric_key.replace('@', '')
                if fallback_key in data:
                    # print(f"Note: Found metric using fallback key '{fallback_key}' in {filepath}") # Less verbose
                    return float(data[fallback_key])
                # Check if the key exists as map@k or recall@k directly (common case from user feedback)
                elif metric_key == 'map@k' and 'map@k' in data:
                     return float(data['map@k'])
                elif metric_key == 'recall@k' and 'recall@k' in data:
                     return float(data['recall@k'])
                else:
                    # print(f"Warning: Metric key '{metric_key}' (or fallbacks) not found in {filepath}") # Less verbose
                    # print(f"Available keys: {list(data.keys())}") # Uncomment for debug
                    return None
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON in {filepath}")
        return None
    except ValueError:
        print(f"Error: Could not convert metric value to float for key '{metric_key}' in {filepath}")
        return None
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

# --- Main Script ---
EVAL_DIR = 'eval_results'
OUTPUT_DIR = 'analysis_plots'
# Regex to capture dataset, metric type (map or recall), k value, q number, t number
FILE_PATTERN = re.compile(r'^(circo|cirr)_(map|recall)(\d+)_ldre_q(\d+)_t(\d+)\.json$')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

results = []
print(f"Scanning directory: {EVAL_DIR}")
all_files = os.listdir(EVAL_DIR)
print(f"Found {len(all_files)} total files/dirs.")

processed_files = 0
extracted_count = 0
for filename in all_files:
    filepath = os.path.join(EVAL_DIR, filename)
    if os.path.isfile(filepath):
        match = FILE_PATTERN.match(filename)
        if match:
            processed_files += 1
            dataset, metric_type, k_value, q_str, t_str = match.groups()
            if dataset == 'cirr':
                metric_type = 'recall'
            # Filter based on dataset and metric type RIGHT HERE
            if (dataset == 'cirr' and metric_type == 'recall') or \
               (dataset == 'circo' and metric_type == 'map'):
                
                try:
                    num_questions = int(q_str)
                    temperature = int(t_str) / 10000.0
                except ValueError:
                    print(f"Warning: Could not parse q/t numbers for {filename}. Skipping.")
                    continue

                # Construct the metric name (e.g., map5, recall10)
                metric_name = f"{metric_type}{k_value}"
                # Construct the key expected in the JSON (e.g., "map@5", "recall@10")
                # Use generic 'map@k' or 'recall@k' if specific k fails
                json_metric_key = f"{metric_type}@{k_value}"
                
                # Extract the metric value
                metric_value = extract_metric_from_json(filepath, json_metric_key)
                
                # Fallback check for generic 'map@k' or 'recall@k' keys if specific one not found
                if metric_value is None:
                    generic_key = f"{metric_type}@k"
                    metric_value = extract_metric_from_json(filepath, generic_key)
                    # if metric_value is not None:
                    #     print(f"Note: Used generic key '{generic_key}' for {filename}")


                if metric_value is not None:
                    extracted_count += 1
                    results.append({
                        'dataset': dataset,
                        'metric': metric_name, # e.g., map5, recall10
                        'num_questions': num_questions,
                        'temperature': temperature,
                        'value': metric_value
                    })
            # else: # File did not match the desired dataset/metric combo
            #     pass 

print(f"Processed {processed_files} files matching the pattern.")
print(f"Extracted data for {extracted_count} relevant points (MAP for CIRCO, Recall for CIRR).")

if not results:
    print("No relevant data extracted. Exiting.")
    exit()

df = pd.DataFrame(results)
print("\nGenerating combined plots (MAP for CIRCO, Recall for CIRR)...")

# --- Combined Plotting ---
for dataset, dataset_df in df.groupby('dataset'):
    
    metric_prefix = "MAP" if dataset == 'circo' else "Recall"
    print(f"--- Generating plots for: {dataset.upper()} ({metric_prefix}@k) ---")

    if dataset_df.empty:
        print("  No data for this dataset. Skipping.")
        continue

    # --- Plot 1: Metrics vs Temperature (faceted by Num Questions) ---
    unique_questions = sorted(dataset_df['num_questions'].unique())
    
    valid_q_facets = []
    for q in unique_questions:
        q_df = dataset_df[dataset_df['num_questions'] == q]
        if q_df.groupby('metric')['temperature'].nunique().max() > 1:
             valid_q_facets.append(q)

    if not valid_q_facets:
        print(f"  Skipping '{dataset.upper()} vs Temp' plot: No facets with multiple points.")
    else:
        num_q_plots = len(valid_q_facets)
        ncols = min(num_q_plots, 3)
        nrows = (num_q_plots + ncols - 1) // ncols
        fig1, axes1 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False, sharey=True)
        axes1 = axes1.flatten()
        plot1_created_count = 0

        for i, q in enumerate(valid_q_facets):
            ax = axes1[i]
            q_df = dataset_df[dataset_df['num_questions'] == q].sort_values('temperature')
            lines_plotted_in_facet = 0
            metrics_in_facet = q_df['metric'].unique()
            # Sort metrics numerically by k (extract number from 'mapX' or 'recallX')
            sorted_metrics = sorted(metrics_in_facet, key=lambda m: int(re.search(r'\d+', m).group()))
            for metric in sorted_metrics:
                metric_df = q_df[q_df['metric'] == metric]
                if len(metric_df) > 1:
                    if 'recall' in metric:
                        metric = metric.replace('recall', 'recall@')
                    if 'map' in metric:
                        metric = metric.replace('map', 'map@')
                    ax.plot(metric_df['temperature'], metric_df['value'], marker='x', linestyle='-', label=metric)
                    lines_plotted_in_facet += 1

            if lines_plotted_in_facet > 0:
                ax.set_title(f'Num Questions = {q}')
                ax.set_xlabel('Temperature')
                if i % ncols == 0:
                    ax.set_ylabel(f'{metric_prefix}@k Value')
                ax.grid(True, linestyle='--', alpha=0.6)
                if lines_plotted_in_facet > 1:
                     ax.legend(fontsize='small')
                plot1_created_count += 1
            else:
                 ax.set_visible(False)

        for j in range(i + 1, len(axes1)):
            fig1.delaxes(axes1[j])

        if plot1_created_count > 0:
            #fig1.suptitle(f'{dataset.upper()} {metric_prefix}@k vs. Temperature', fontsize=16, y=0.99)
            fig1.tight_layout(rect=[0, 0.03, 1, 0.96])
            plot1_filename = os.path.join(OUTPUT_DIR, f'{dataset}_{metric_prefix}_vs_temp_combined.pdf')
            try:
                fig1.savefig(plot1_filename)
                print(f"  Saved: {plot1_filename}")
            except Exception as e:
                print(f"  Error saving plot {plot1_filename}: {e}")
        plt.close(fig1)

    # --- Plot 2: Metrics vs Num Questions (faceted by Temperature) ---
    unique_temps = sorted(dataset_df['temperature'].unique())

    valid_t_facets = []
    for t in unique_temps:
        t_df = dataset_df[dataset_df['temperature'] == t]
        if t_df.groupby('metric')['num_questions'].nunique().max() > 1:
            valid_t_facets.append(t)
            
    max_facets = 12
    if len(valid_t_facets) > max_facets:
        indices = np.round(np.linspace(0, len(valid_t_facets) - 1, max_facets)).astype(int)
        selected_temps = [valid_t_facets[i] for i in sorted(list(set(indices)))]
        print(f"  (Plotting subset of {len(selected_temps)} temperatures out of {len(valid_t_facets)} valid facets)")
    else:
        selected_temps = valid_t_facets

    if not selected_temps:
         print(f"  Skipping '{dataset.upper()} vs Questions' plot: No facets with multiple points.")
    else:
        num_t_plots = len(selected_temps)
        ncols = min(num_t_plots, 3)
        nrows = (num_t_plots + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False, sharey=True)
        axes2 = axes2.flatten()
        plot2_created_count = 0

        for i, t in enumerate(selected_temps):
            ax = axes2[i]
            t_df = dataset_df[dataset_df['temperature'] == t].sort_values('num_questions')
            lines_plotted_in_facet = 0
            metrics_in_facet = t_df['metric'].unique()
            sorted_metrics = sorted(metrics_in_facet, key=lambda m: int(re.search(r'\d+', m).group()))

            for metric in sorted_metrics:
                metric_df = t_df[t_df['metric'] == metric]
                if len(metric_df) > 1:
                    if 'recall' in metric:
                        metric = metric.replace('recall', 'recall@')
                    if 'map' in metric:
                        metric = metric.replace('map', 'map@')
                    ax.plot(metric_df['num_questions'], metric_df['value'], marker='x', linestyle='-', label=metric)
                    lines_plotted_in_facet += 1

            if lines_plotted_in_facet > 0:
                ax.set_title(f'Temperature = {t:.4f}')
                ax.set_xlabel('Num Questions')
                if i % ncols == 0:
                    ax.set_ylabel(f'{metric_prefix}@k Value')
                ax.grid(True, linestyle='--', alpha=0.6)
                if lines_plotted_in_facet > 1:
                    ax.legend(fontsize='small')
                plot2_created_count += 1
            else:
                 ax.set_visible(False)

        for j in range(i + 1, len(axes2)):
            fig2.delaxes(axes2[j])

        if plot2_created_count > 0:
            #fig2.suptitle(f'{dataset.upper()} {metric_prefix}@k vs. Num Questions', fontsize=16, y=0.99)
            fig2.tight_layout(rect=[0, 0.03, 1, 0.96])
            plot2_filename = os.path.join(OUTPUT_DIR, f'{dataset}_{metric_prefix}_vs_q_combined.pdf')
            try:
                fig2.savefig(plot2_filename)
                print(f"  Saved: {plot2_filename}")
            except Exception as e:
                 print(f"  Error saving plot {plot2_filename}: {e}")
        plt.close(fig2)

print("\nAnalysis complete. Combined plots (MAP for CIRCO, Recall for CIRR) saved in:", OUTPUT_DIR)
