import os
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def calculate_statistics(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        epoch = []
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []

        for row in reader:
            epoch.append(int(row['Epoch']))
            training_loss.append(float(row['Training Loss']))
            training_accuracy.append(float(row['Training Accuracy']))
            validation_loss.append(float(row['Validation Loss']))
            validation_accuracy.append(float(row['Validation Accuracy']))

    statistics = {
        'Epoch': epoch,
        'Training Loss': training_loss,
        'Training Accuracy': training_accuracy,
        'Validation Loss': validation_loss,
        'Validation Accuracy': validation_accuracy
    }

    return statistics

def main():
    directory = r'.\results\cnnLogs'

    # Dictionaries to store accumulated results
    accumulated_results = {'Training Loss': [], 'Training Accuracy': [], 'Validation Loss': [], 'Validation Accuracy': []}

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)

            statistics = calculate_statistics(file_path)

            for key in accumulated_results.keys():
                accumulated_results[key].append(statistics[key])

    # Calculate mean and standard deviation for each metric
    for key in ['Training Loss', 'Training Accuracy']:
        mean_value = np.mean(accumulated_results[key], axis=0)
        std_deviation = np.std(accumulated_results[key], axis=0)

        print('{}:'.format(key))
        print('  Mean={:.4f}, Standard Deviation={:.4f}'.format(np.mean(mean_value), np.mean(std_deviation)))

        # Create point plot with error bars using matplotlib
        plt.figure()
        plt.errorbar(statistics['Epoch'], mean_value, yerr=std_deviation, fmt='o', capsize=5, label='Data Points')
        plt.plot(statistics['Epoch'], mean_value, label='Mean Line', color='red')  # Add the line
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.title(f'{key} - Mean and Standard Deviation')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
