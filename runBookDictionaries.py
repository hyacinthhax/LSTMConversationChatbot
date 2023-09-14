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
import nltk
from nltk.tokenize import sent_tokenize
from chatbotTrainer import ChatbotTrainer  # Import your ChatbotTrainer class

nltk.download('punkt')  # Download the sentence tokenizer model if not already installed

def book_to_dict(book_filename):
    print("Making Dictionary...  ")
    book_dict = {}  # Initialize an empty dictionary to store the book data
    with open(book_filename, 'r', encoding='utf-8') as f:
        data = f.read()

    # Tokenize the text into sentences
    sentences = sent_tokenize(data)

    # Iterate through sentences to create pairs
    pairs = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]

    # Create the dictionary with book title as the key
    book_title = book_filename.split('.')[0]  # Extract the title from the filename
    book_dict[book_title] = pairs

    return book_dict

def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    chatbot_trainer = ChatbotTrainer()

    # Initialize the tokenizer
    if os.path.exists(chatbot_trainer.tokenizer_save_path):
        with open(chatbot_trainer.tokenizer_save_path, 'rb') as tokenizer_load_file:
            chatbot_trainer.tokenizer = pickle.load(tokenizer_load_file)
            chatbot_trainer.tokenizer.num_words = chatbot_trainer.max_vocab_size
            chatbot_trainer.logger.info("Model and tokenizer loaded successfully.")
    else:
        print("Tokenizer not found, making now...  ")
        chatbot_trainer.tokenizer = Tokenizer(oov_token="<OOV>", num_words=chatbot_trainer.max_vocab_size)  # Initialize the Tokenizer
        chatbot_trainer.tokenizer.num_words = chatbot_trainer.max_vocab_size

    # Prompt the user for the book name
    bookName = input("(Put in folder with runBookDictionaries.py)\n(SomeBookCalled.txt)\n> ")

    # Process book data and create the dialog_data dictionary
    dialog_data = book_to_dict(bookName)
    
    # Once all book data is processed, you can fit the tokenizer
    all_input_texts = [chatbot_trainer.preprocess_text(pair[0]) for pairs in dialog_data.values() for pair in pairs]
    all_target_texts = [chatbot_trainer.preprocess_text(pair[1]) for pairs in dialog_data.values() for pair in pairs]
    
    # Fit the tokenizer on all book data
    chatbot_trainer.tokenizer.fit_on_texts(all_input_texts + all_target_texts)

    # Train models for each book
    for bookName, sentences in dialog_data.items():
        # Load the model
        chatbot_trainer.load_model()

        # Separate the input and target texts
        input_texts = [chatbot_trainer.preprocess_text(pair[0]) for pair in sentences]
        target_texts = [chatbot_trainer.preprocess_text(pair[1]) for pair in sentences]

        # Split data into train and test for this book
        train_input, test_input, train_target, test_target = train_test_split(
            input_texts, target_texts, test_size=0.2, random_state=42)

        # Check if there are enough dialogue pairs for training
        if len(train_input) < 2 or len(train_target) < 2:
            chatbot_trainer.logger.warning(f"Skipping training for sentence in {bookName} due to insufficient training data.")
            continue

        # Train the model using the training data for this book
        conversation_id = f"'{bookName}'"
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

        chatbot_trainer.logger.info(f"Test loss for Book; {bookName}: {test_loss}")
        chatbot_trainer.logger.info(f"Test accuracy for Book; {bookName}: {test_accuracy}")

        # Save the model
        chatbot_trainer.save_model()

    recent_user_input = None
    recent_chatbot_response = None

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Print the most recent chatbot response
        if recent_chatbot_response:
            print(f"Chatbot: {recent_chatbot_response}")

        # Generate and print the new response
        response = chatbot_trainer.generate_response(user_input)
        print(f"Chatbot: {response}")

        # Update context
        recent_user_input = user_input
        recent_chatbot_response = response

if __name__ == "__main__":
    main()
