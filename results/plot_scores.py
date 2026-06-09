import csv
import os
import random
import matplotlib.pyplot as plt

# Resolving absolute paths based on the known project structure
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

if os.path.basename(current_dir) == 'results':
    base_dir = os.path.dirname(current_dir)
elif os.path.basename(current_dir) == 'nglab_llm_hallu':
    base_dir = current_dir
else:
    base_dir = '/mnt/c/Users/gideo/nglab/nglab_llm_hallu'

justice_path = os.path.join(base_dir, 'datasets', 'justice_final_results.csv')

def create_scatter_plot(x_coords, y_coords, out_path, x_label, color):
    plt.figure(figsize=(10, 5))
    plt.scatter(x_coords, y_coords, c=color, alpha=0.5, edgecolors='black')
    plt.xlabel(x_label)
    plt.ylabel('Actual Label')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.2, 1.2)
    plt.yticks([0, 1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved PNG graphic to: {out_path}")

def generate_plots():
    results_dir = os.path.join(base_dir, 'results')
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    print("Available .csv files in results/:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}. {file}")
    
    choice = input("Enter the number or filename of the .csv file to use for averages: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
        selected_csv = csv_files[int(choice) - 1]
    elif choice in csv_files:
        selected_csv = choice
    elif choice + '.csv' in csv_files:
        selected_csv = choice + '.csv'
    else:
        print("Invalid choice, defaulting to test_results.csv")
        selected_csv = "test_results.csv"
        
    test_results_path = os.path.join(results_dir, selected_csv)
    print(f"Using {selected_csv} for test scores...")

    test_scores = {}
    with open(test_results_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                test_scores[row['post_id']] = float(row['average_score'])
            except ValueError:
                pass

    actual_labels = {}
    gat_scores = {}
    with open(justice_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['post_id'].replace('t3_', '')
            try:
                actual_labels[pid] = int(row['actual_label'])
                gat_scores[pid] = float(row['gat_confidence_score'])
            except ValueError:
                pass

    x_scores = []
    y_labels = []
    combined_data = []
    for pid, avg_score in test_scores.items():
        if pid in actual_labels:
            x_scores.append(avg_score)
            y_labels.append(actual_labels[pid])
            combined_data.append({'post_id': pid, 'average_score': avg_score, 'actual_label': actual_labels[pid]})
            
    combined_csv_path = os.path.join(results_dir, 'average_actual_combined.csv')
    with open(combined_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['post_id', 'average_score', 'actual_label'])
        writer.writeheader()
        writer.writerows(combined_data)
    print(f"Saved combined data to: {combined_csv_path}")
            
    x_gat = []
    y_gat_labels = []
    for pid, gat_score in gat_scores.items():
        if pid in actual_labels and pid in test_scores:
            x_gat.append(gat_score)
            y_gat_labels.append(actual_labels[pid])

    out1 = os.path.join(base_dir, 'results', 'average_vs_actual_plot.png')
    create_scatter_plot(x_scores, y_labels, out1, 'Average Score', 'blue')
    
    out2 = os.path.join(base_dir, 'results', 'gat_vs_actual_plot.png')
    create_scatter_plot(x_gat, y_gat_labels, out2, 'GAT Confidence Score', 'teal')

if __name__ == "__main__":
    generate_plots()
