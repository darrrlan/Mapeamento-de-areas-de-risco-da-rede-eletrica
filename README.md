# 🌳 Mapeamento de Áreas de Risco da Rede Elétrica

Este projeto utiliza Redes Neurais Convolucionais (CNN) para identificar automaticamente áreas de risco em redes elétricas causadas pela arborização urbana. A proposta visa auxiliar na manutenção preventiva, reduzir falhas no fornecimento de energia e aumentar a segurança do sistema elétrico.

## 📌 Objetivo

Desenvolver um modelo de aprendizado profundo capaz de classificar imagens de vegetação urbana como "Alto Risco" ou "Baixo Risco" de interferência com a rede elétrica.

## 🧠 Tecnologias e Metodologia

- Python + TensorFlow/Keras
- CNN (Convolutional Neural Networks)
- Dataset personalizado com imagens rotuladas
- Pré-processamento com conversão para escala de cinza
- Classificação binária com função de ativação sigmoide

### Arquitetura da CNN

| Camada         | Parâmetros       | Ativação |
|----------------|------------------|----------|
| Conv2D         | 32 filtros (3x3) | ReLU     |
| MaxPooling2D   | (2x2)            | -        |
| Conv2D         | 64 filtros (3x3) | ReLU     |
| MaxPooling2D   | (2x2)            | -        |
| Conv2D         | 128 filtros (3x3)| ReLU     |
| Flatten        | -                | -        |
| Dense          | 128 unidades     | ReLU     |
| Dense          | 1 unidade        | Sigmóide |

## 🗂 Dataset

- Total de 1328 imagens (128x128 px)
  - 660 imagens: Alto risco
  - 668 imagens: Baixo risco
- Imagens coletadas manualmente
- Dataset balanceado
- Disponível em: [github.com/darrrlan/Mapeamento-de-areas-de-risco-da-rede-eletrica](https://github.com/darrrlan/Mapeamento-de-areas-de-risco-da-rede-eletrica)

## 📊 Resultados

- Acurácia de ~98% nos testes
- AUC (Área sob a curva ROC) próxima de 1
- Convergência rápida: 5 épocas já apresentam bons resultados
- Pré-processamento em escala de cinza melhora a eficiência sem prejudicar a acurácia

## 🚧 Desafios e Melhorias Futuras

- Melhorar classificação de árvores podadas
- Reduzir confusão entre postes e árvores
- Ampliar o dataset com mais imagens em diferentes condições
- Explorar novos ângulos e cenários urbanos

## 🚀 Como Executar

```bash
# Clone o repositório
git clone https://github.com/darrrlan/Mapeamento-de-areas-de-risco-da-rede-eletrica.git
cd Mapeamento-de-areas-de-risco-da-rede-eletrica

# (Recomenda-se usar ambiente virtual)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Execute o script principal
python main.py
```

---

Desenvolvido por Darlan Oliveira – UTFPR – Apucarana
