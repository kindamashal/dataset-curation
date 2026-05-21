import json
import os
import matplotlib.pyplot as plt
import string

FILES_TO_PLOT = [
    # r"model_outputs\intervened\red_invert_a100_foi.json",
    # r"model_outputs\intervened\green_invert_a100_foi.json",
    # r"model_outputs\intervened\blue_steer_red_a100_foi.json",
    # r"model_outputs\intervened\red_ablate_a0_foi.json",
    # r"model_outputs\intervened\blue_ablate_a0_foi.json",
    # r"model_outputs\intervened\green_ablate_a0_foi.json",
    r"model_outputs\intervened\red_steer_blue_a100_foi.json",
    r"model_outputs\intervened\red_steer_green_a500_foi.json",
    r"model_outputs\intervened\blue_steer_red_a500_foi.json",
    r"model_outputs\intervened\blue_steer_green_a500_excl_rb.json",
    r"model_outputs\intervened\green_steer_red_a100_foi.json",
    r"model_outputs\intervened\green_steer_blue_a100_foi.json"
    
]

OUTPUT_CHART_NAME = r"model_outputs\intervened\plots\steering_results.png"
CHART_TITLE = "Effects of Steering"

def calculate_accuracy(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0.0
        
    correct = 0
    total = len(data)
    
    if total == 0:
        return 0.0
        
    for item in data:
        output = str(item.get('intervened_output', '')).lower()
        answer = str(item.get('correct_answer', '')).lower()
        
        output = output.translate(str.maketrans('', '', string.punctuation)).strip()
        answer = answer.translate(str.maketrans('', '', string.punctuation)).strip()
        
        if answer and answer in output:
            correct += 1
            
    return (correct / total) * 100

def main():
    labels = []
    accuracies = []
    
    for file_path in FILES_TO_PLOT:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        acc = calculate_accuracy(file_path)
        
        name = os.path.splitext(os.path.basename(file_path))[0]
        
        labels.append(name)
        accuracies.append(acc)
        
    if not labels:
        print("No valid files processed. Exiting.")
        return

    sorted_pairs = sorted(zip(labels, accuracies), key=lambda x: x[1], reverse=True)
    labels, accuracies = zip(*sorted_pairs)
    labels = list(labels)
    accuracies = list(accuracies)

    color_map = {
        'red': 'tomato',
        'blue': 'dodgerblue',
        'green': 'limegreen',
    }
    
    bar_colors = []
    for label in labels:
        found_color = 'gray'
        label_lower = label.lower()
        if label_lower.startswith('red'):
            found_color = color_map['red']
        elif label_lower.startswith('blue'):
            found_color = color_map['blue']
        elif label_lower.startswith('green'):
            found_color = color_map['green']
        bar_colors.append(found_color)
        
    # Plotting
    plt.figure(figsize=(12, 6))
    bars = plt.barh(labels, accuracies, color=bar_colors, edgecolor='black', alpha=0.9)
    
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title(CHART_TITLE, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=10)
    plt.xlim(0, 100)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART_NAME, dpi=300)
    print(f"Successfully generated horizontal bar chart and saved to: {OUTPUT_CHART_NAME}")

if __name__ == "__main__":
    main()