import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import pickle
import convokit
from processed_dialogs import dialog_data  # Import the dialog_data dictionary
from playsound import playsound
from chatbotTrainer import ChatbotTrainer  # Import your ChatbotTrainer class


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    chatbot_trainer = ChatbotTrainer()

    # Initialize the corpus
    corpus_path = "C:\\Users\\admin\\Desktop\\movie-corpus"
    if os.path.exists(chatbot_trainer.tokenizer_save_path):
        with open(chatbot_trainer.tokenizer_save_path, 'rb') as tokenizer_load_file:
            chatbot_trainer.tokenizer = pickle.load(tokenizer_load_file)
            chatbot_trainer.tokenizer.num_words = chatbot_trainer.max_vocab_size
            chatbot_trainer.logger.info("Model and tokenizer loaded successfully.")
            chatbot_trainer.load_corpus(corpus_path)  # Use the load_corpus method to load the corpus
    else:
        print("Tokenizer not found, making now...  ")
        chatbot_trainer.tokenizer = Tokenizer(oov_token="<OOV>", num_words=chatbot_trainer.max_vocab_size)  # Initialize the Tokenizer
        chatbot_trainer.tokenizer.num_words = chatbot_trainer.max_vocab_size
        chatbot_trainer.load_corpus(corpus_path)  # Use the load_corpus method to load the corpus

    # Once all speakers' data is processed, you can fit the tokenizer
    all_input_texts = [chatbot_trainer.preprocess_text(pair[0]) for pairs in dialog_data.values() for pair in pairs]
    all_target_texts = [chatbot_trainer.preprocess_text(pair[1]) for pairs in dialog_data.values() for pair in pairs]
    train_input_texts, test_input_texts, train_target_texts, test_target_texts = train_test_split(all_input_texts, all_target_texts, test_size=0.2, random_state=42)

    chatbot_trainer.tokenizer.fit_on_texts(train_input_texts + train_target_texts)

    # Train models for each speaker
    for speaker, speaker_dialogue_pairs in dialog_data.items():
        # Load the model
        chatbot_trainer.load_model()

        # Separate the input and target texts
        input_texts = [chatbot_trainer.preprocess_text(pair[0]) for pair in speaker_dialogue_pairs]
        target_texts = [chatbot_trainer.preprocess_text(pair[1]) for pair in speaker_dialogue_pairs]

        # Split data into train and test for this speaker
        train_input, test_input, train_target, test_target = train_test_split(
            input_texts, target_texts, test_size=0.2, random_state=42)

        # Check if there are enough dialogue pairs for training
        if len(train_input) < 2 or len(train_target) < 2:
            chatbot_trainer.logger.warning(f"Skipping training for Conversation {speaker} due to insufficient training data.")
            continue

        # Train the model using the training data for this speaker
        conversation_id = f"'{speaker}'"
        history = chatbot_trainer.train_model(train_input, train_target, conversation_id)

        # Preprocess the test input data using the tokenizer
        test_input_sequences = chatbot_trainer.tokenizer.texts_to_sequences(test_input)
        padded_test_input_sequences = pad_sequences(test_input_sequences, maxlen=chatbot_trainer.max_seq_length, padding='post')

        # Preprocess the test target data using the tokenizer
        test_target_sequences = chatbot_trainer.tokenizer.texts_to_sequences(test_target)
        padded_test_target_sequences = pad_sequences(test_target_sequences, maxlen=chatbot_trainer.max_seq_length, padding='post')

        # Evaluate the model on the preprocessed test data
        test_loss, test_accuracy = chatbot_trainer.model.evaluate(
            [padded_test_input_sequences, padded_test_target_sequences],
            padded_test_target_sequences,
            batch_size=chatbot_trainer.batch_size)

        chatbot_trainer.logger.info(f"Test loss for Conversation {speaker}: {test_loss}")
        chatbot_trainer.logger.info(f"Test accuracy for Conversation {speaker}: {test_accuracy}")

        # Save the model
        chatbot_trainer.save_model()



if __name__ == "__main__":
    main()
