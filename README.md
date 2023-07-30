# AcrTransAct

Welcome to the AcrTransAct repository! You can find our online tool here: (https://acrtransact.usask.ca).

This repository serves as a comprehensive method to reproduce the experiments conducted in our paper titled: **AcrTransAct: Pre-trained Protein Transformer Models for the Detection of Type I Anti-CRISPR Activities** and provides insight into the backend of our tool.

By sharing the source code and data, we aim to foster transparency and facilitate understanding for anyone interested in exploring and validating the results obtained through our work. Feel free to utilize this repository to replicate our experiments and delve deeper into the functionalities of our tool.

## Overview

AcrTransAct is a Python-based tool developed for predicting the inhibition of CRISPR-Cas systems by anti-CRISPR proteins (Acrs). It utilizes deep learning models, specifically Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks, to predict interactions between Acrs and CRISPR-Cas complexes, collectively contributing to inhibition. The tool takes protein sequences and structural features as input and provides accurate predictions of CRISPR-Cas inhibition.

## Features

- Predict CRISPR-Cas inhibition using Acr sequences and structural features.
- Supports both CNN and LSTM models for prediction.
- Provides evaluation metrics such as F1 score, accuracy, and AUC.

## Requirements

Python 3.9
bertviz==1.4.0
Bio==1.5.9
biolib==0.1.9
biopython==1.80
ipython==8.3.0
matplotlib==3.5.2
numpy==1.21.5
pandas==1.4.2
pytorch_lightning==1.8.5.post0
scikit_learn==1.2.2
seaborn==0.11.2
torch==1.12.0
transformers==4.24.0

* Optional:
wandb

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/USask-BINFO/AcrTransAct.git
cd AcrTransAct
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can run the bash file under the code/scripts called train.sh to train all of the models in our work.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We would like to thank all the contributors and researchers who have contributed to the development of AcrTransAct.

## Contact

For any questions or inquiries, please open an issue on our repository or contact us at [moeinhasani77@gmail.com](mailto:moeinhasani77@gmail.com).
