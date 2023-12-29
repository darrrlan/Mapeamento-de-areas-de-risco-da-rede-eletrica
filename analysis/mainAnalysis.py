import os
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


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

def visualize_activation_maps(model, image):
    # Criando modelo para obter saídas intermediárias
    layer_outputs = [layer.output for layer in model.layers[:-1]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(np.expand_dims(image, axis=0))

    # Visualização da entrada, ativações de cada camada e saída final
    num_layers = len(activations)
    num_cols = 4
    num_rows = (num_layers + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 8), dpi=500)

    # Imagem de entrada
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(image)
    plt.title('Input Image', fontsize=5)
    plt.axis('off')

    # Visualização das ativações de cada camada convolucional
    for i in range(num_layers):
        activation_maps = activations[i][0]  # Get the activation map for the first (and only) sample
        num_filters = activation_maps.shape[-1]

        # Mostrar todos os mapas de ativação em uma grade
        for j in range(num_filters):
            subplot_num = i * num_cols + j + 2
            if subplot_num <= num_layers:
                plt.subplot(num_rows, num_cols, subplot_num)
                plt.imshow(activation_maps[:, :, j], cmap='viridis')
                plt.title(f'Conv Layer {i + 1}, Filter {j + 1}', fontsize=5)  # Adjust font size
                plt.axis('off')
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labelleft=False, labelsize=8)  # Adjust tick label font size

    plt.tight_layout()
    plt.show()


def load_and_process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check the number of channels
    num_channels = image.shape[2] if len(image.shape) == 3 else 1

    # Convert to RGB or grayscale accordingly
    if num_channels == 3:
        # RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1) / 255.0

    # Resize the image
    image = cv2.resize(image, (128, 128))

    return image

def generate_confusion_matrix(predictions_path, test_labels_path):
    # Load the saved predictions and test labels
    predictions = np.load(predictions_path)
    test_labels = np.load(test_labels_path)

    # Create a confusion matrix
    confusion_mat = confusion_matrix(test_labels, predictions)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Low risk', 'High risk'], yticklabels=['Low risk', 'High risk'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def generate_roc_curve(predictions_probs_path, test_labels_path):
    # Load the saved predictions and test labels
    predictions_probs = np.load(predictions_probs_path)
    test_labels = np.load(test_labels_path)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(test_labels, predictions_probs)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def exibir_imagens_predicoes_0_e_1(test_images_path, test_labels, predictions, num_imagens=2):
    # Load the test images (assuming they are file paths)
    test_images = [load_and_process_image(image_path) for image_path in np.load(test_images_path)]

    indices_1_para_0 = np.where((test_labels == 1) & (predictions.flatten() == 0))[0]
    indices_0_para_1 = np.where((test_labels == 0) & (predictions.flatten() == 1))[0]

    # Determine o número máximo de imagens em uma linha
    tam_0_para_1 = min(2, len(indices_0_para_1))
    tam_1_para_0 = min(2, len(indices_1_para_0))
    max_images_per_row = max(tam_1_para_0, tam_0_para_1, num_imagens)

    # Exibir até 'num_imagens' imagens para cada condição
    num_rows = 2
    num_cols = min(max_images_per_row, 4)  # Máximo de 4 colunas para melhorar o layout

    plt.figure(figsize=(15, 8))

    # Exibir imagens para a condição de 1 para 0
    for i, idx_1_para_0 in enumerate(indices_1_para_0[:num_imagens]):
        # Check if the index is within bounds
        if idx_1_para_0 < len(test_images):
            img_1_para_0 = test_images[idx_1_para_0]

            # Converte para RGB se não estiver em RGB
            if len(img_1_para_0.shape) == 2:
                img_1_para_0 = plt.cm.gray(img_1_para_0)

            verdadeiro_rotulo_1_para_0 = "Alto Risco" if test_labels[idx_1_para_0] == 1 else "Baixo Risco"
            predito_rotulo_1_para_0 = "Alto Risco" if predictions[idx_1_para_0] == 0 else "Baixo Risco"

            # Subplot para 1 para 0
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img_1_para_0)
            plt.title(f"Verdadeiro: {verdadeiro_rotulo_1_para_0}\nPrevisto: {predito_rotulo_1_para_0}")
            plt.axis('off')

    # Exibir imagens para a condição de 0 para 1
    for i, idx_0_para_1 in enumerate(indices_0_para_1[:num_imagens]):
        # Check if the index is within bounds
        if idx_0_para_1 < len(test_images):
            img_0_para_1 = test_images[idx_0_para_1]

            # Converte para RGB se não estiver em RGB
            if len(img_0_para_1.shape) == 2:
                img_0_para_1 = plt.cm.gray(img_0_para_1)

            verdadeiro_rotulo_0_para_1 = "Alto Risco" if test_labels[idx_0_para_1] == 1 else "Baixo Risco"
            predito_rotulo_0_para_1 = "Alto Risco" if predictions[idx_0_para_1] == 1 else "Baixo Risco"

            # Subplot para 0 para 1
            plt.subplot(num_rows, num_cols, i + num_imagens + 1)
            plt.imshow(img_0_para_1)
            plt.title(f"Verdadeiro: {verdadeiro_rotulo_0_para_1}\nPrevisto: {predito_rotulo_0_para_1}")
            plt.axis('off')

    # Ajustes gerais para layout
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Ajusta os espaçamentos horizontal e vertical
    plt.show()






# Exibir imagens para as previsões onde o valor verdadeiro é 1 e o valor previsto é 0, e vice-versa





def main():
    directory = r'.\results\cnnLogs'

    print("Diretório de Trabalho Atual:", os.getcwd())
    print("Arquivos no Diretório:", os.listdir())

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

        '''

        # Create point plot with error bars using matplotlib
        plt.figure()
        plt.errorbar(statistics['Epoch'], mean_value, yerr=std_deviation, fmt='o', capsize=5, label='Data Points')
        plt.plot(statistics['Epoch'], mean_value, label='Mean Line', color='red')  # Add the line
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.title(f'{key} - Mean and Standard Deviation')
        plt.legend()
        plt.show()
        '''


    # Load the model and weights
    model_path = 'analysis/CNN_model.keras'
    weights_path = 'analysis/CNN_weights.keras'
    model = load_model(model_path)
    model.load_weights(weights_path)

    # Load and process the image for visualization
    image_path = 'dataset_vegetation_on_electrical_grid/Alto risco/2.png'
    image = load_and_process_image(image_path)

    '''
    # Visualize activation maps
    visualize_activation_maps(model, image)
    '''

    # Specify the paths for predictions and test labels
    predictions_path = 'analysis/predictions.npy'
    test_labels_path = 'analysis/test_labels.npy'
    test_images_path = 'analysis/test_image_paths.npy'

    '''
    # Generate and plot the confusion matrix
    generate_confusion_matrix(predictions_path, test_labels_path)
     # Generate and plot the ROC curve
    generate_roc_curve(predictions_path, test_labels_path)

    '''
    test_predictions = np.load(predictions_path)
    test_labels = np.load(test_labels_path)
    exibir_imagens_predicoes_0_e_1(test_images_path, test_labels, test_predictions, num_imagens=2)

if __name__ == "__main__":
    main()