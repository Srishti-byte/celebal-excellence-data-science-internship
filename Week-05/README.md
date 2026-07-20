# Week 5 – Text Generation using Vanilla RNN, LSTM and GRU

## Objective

The objective of this project is to build and compare recurrent neural network architectures for next-word prediction using an English translation of the Bhagavad Gita. The project evaluates the performance of Vanilla RNN, LSTM, and GRU models and studies the effect of hyperparameter tuning on text generation.

## Dataset

The dataset consists of the English translation of the **Bhagavad Gita**, containing continuous textual data used for sequence modeling and next-word prediction.

## Tasks Performed

- Text preprocessing and data cleaning
- Tokenization and vocabulary creation
- Text-to-sequence conversion
- N-gram sequence generation
- Sequence padding
- Creation of input features and target labels
- Vanilla RNN implementation
- LSTM implementation
- GRU implementation
- Training loss comparison
- Next-word text generation
- Hyperparameter tuning
- Performance comparison between baseline and improved models

## Results

- Successfully implemented Vanilla RNN, LSTM, and GRU models for next-word prediction.
- All three models learned meaningful sequential patterns from the text corpus.
- Improved models achieved lower training loss after increasing the embedding dimension, hidden units, and training epochs.
- The improved LSTM achieved the lowest final training loss among the three architectures.
- The generated text demonstrated that recurrent neural networks can effectively learn contextual word relationships from sequential text data.

## Technologies Used

- Python
- NumPy
- Matplotlib
- TensorFlow / Keras
- Google Colab
- Kaggle

## Files

- `Week5_SrishtiGupta.ipynb` – Complete implementation of the project.
- `bhagavad_gita.txt` – Dataset used for training and text generation.
