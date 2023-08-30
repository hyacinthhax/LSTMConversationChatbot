import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import convokit
from processed_dialogs import dialog_data  # Import the dialog_data dictionary
from playsound import playsound


class ChatbotTrainer:
    def __init__(self):
        self.corpus = None
        self.max_vocab_size = max_vocab_size = 10000
        self.model = None
        self.model_filename = "chatbot_model.h5"
        self.tokenizer = None
        self.logger = self.setup_logger()  # Initialize your logger here
        self.embedding_dim = 50  # Define the embedding dimension here
        self.max_seq_length = 100  # Replace with your desired sequence length
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epochs = 10
        self.lstm_units = 128


    def plot_and_save_training_metrics(self, history, speaker):
        # Plot training metrics such as loss and accuracy
        plt.figure(figsize=(10, 6))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save the plot as an image file
        plot_filename = f"C:\\Users\\admin\\Desktop\\ChatBotMetrics\\{speaker}_training_metrics.png"
        plt.tight_layout()
        plt.savefig(plot_filename)  # Save the plot as an image
        plt.close()  # Close the plot to free up memory

        return plot_filename


    def load_corpus(self, corpus_path):
        self.logger.info("Loading and preprocessing corpus...")
        self.corpus = convokit.Corpus(filename=corpus_path)
        self.logger.info("Corpus loaded and preprocessed successfully.")

    def setup_logger(self):
        logger = logging.getLogger("ChatbotTrainer")
        logger.setLevel(logging.DEBUG)

        # Create console handler and set level to INFO for progress reports
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Create a file handler and set level to DEBUG for progress reports and ERROR for error notifications
        file_handler = logging.FileHandler("chatbot.log")
        file_handler.setLevel(logging.DEBUG)  # Set level to DEBUG to capture progress reports
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def preprocess_text(text):
        # Remove double quotes from the text
        cleaned_text = text.replace('"', '')

        # Remove multiple spaces
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text.lower()


    def build_model(self):
        self.logger.info("Building model...")
        lstm_units = self.lstm_units
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")

        vocab_size = len(self.tokenizer.word_index) + 1  # +1 for padding token
        
        max_seq_length = self.max_seq_length
        learning_rate = self.learning_rate

        # Encoder
        encoder_inputs = Input(shape=(max_seq_length,))
        encoder_embedding = Embedding(input_dim=vocab_size, output_dim=self.embedding_dim)(encoder_inputs)
        encoder_lstm = LSTM(units=lstm_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(max_seq_length,))
        decoder_embedding = Embedding(input_dim=vocab_size, output_dim=self.embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(units=lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(units=vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Create the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.logger.info("Model built successfully.")


    def train_model(self, input_texts, target_texts):
        self.logger.info("Training Model...")

        if self.corpus is None or self.tokenizer is None:
            raise ValueError("Corpus or tokenizer is not initialized.")

        # Fit the tokenizer on the combined input and target texts
        all_texts = input_texts + target_texts
        self.tokenizer.fit_on_texts(all_texts)

        # Tokenize input and target texts
        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)

        max_seq_length = self.max_seq_length  # Specify your maximum sequence length
        padded_input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
        padded_target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

        # Build the model (if not already built)
        if self.model is None:
            self.build_model()

        # Compile the model (no need to specify learning_rate here, it's done in build_model)
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy
                           metrics=['accuracy'])

        # Train the model
        history = self.model.fit(
            [padded_input_sequences, padded_target_sequences],
            padded_target_sequences,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2
        )

        # Log training metrics
        self.logger.info("Training metrics:")
        for key, value in history.history.items():
            self.logger.info(f"{key}: {value}")

        self.logger.info("Model trained successfully.")
        self.save_model()

        return history  # Return the history object


    def save_model(self):
        self.logger.info("Saving Model...")
        if self.model:
            self.model.save(self.model_filename)
        else:
            self.logger.warning("No model to save.")

    def load_model(self):
        self.logger.info("Loading Model...")
        if os.path.exists(self.model_filename):
            self.model = tf.keras.models.load_model(self.model_filename)
            self.logger.info("Model loaded successfully.")
        else:
            self.logger.warning("No saved model found.")


if __name__ == "__main__":
    chatbot_trainer = ChatbotTrainer()

    # Initialize the corpus
    corpus_path = "C:\\Users\\admin\\Desktop\\movie-corpus"
    chatbot_trainer.corpus = convokit.Corpus(filename=corpus_path)

    chatbot_trainer.tokenizer = Tokenizer(oov_token="<OOV>", num_words=chatbot_trainer.max_vocab_size)  # Initialize the Tokenizer

    # Once all speakers' data is processed, you can fit the tokenizer
    all_input_texts = [pair[0] for pairs in dialog_data.values() for pair in pairs]
    all_target_texts = [pair[1] for pairs in dialog_data.values() for pair in pairs]
    train_input_texts, test_input_texts, train_target_texts, test_target_texts = train_test_split(all_input_texts, all_target_texts, test_size=0.2, random_state=42)

    chatbot_trainer.tokenizer.fit_on_texts(train_input_texts + train_target_texts)

    # Train and save models for each speaker
    for speaker, speaker_dialogue_pairs in dialog_data.items():
        # Load the model
        chatbot_trainer.load_model()
        # Separate the input and target texts
        input_texts = [pair[0] for pair in speaker_dialogue_pairs]
        target_texts = [pair[1] for pair in speaker_dialogue_pairs]

        # Split data into train and test for this speaker
        train_input, test_input, train_target, test_target = train_test_split(
            input_texts, target_texts, test_size=0.2, random_state=42)

        # Check if there are enough dialogue pairs for training
        if len(train_input) < 2 or len(train_target) < 2:
            chatbot_trainer.logger.warning(f"Skipping training for speaker {speaker} due to insufficient training data.")
            continue

        # Train the model using the training data for this speaker
        history = chatbot_trainer.train_model(train_input, train_target)

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

        chatbot_trainer.logger.info(f"Test loss for speaker {speaker}: {test_loss}")
        chatbot_trainer.logger.info(f"Test accuracy for speaker {speaker}: {test_accuracy}")

        # Save the model
        chatbot_trainer.save_model()

        # Save training metrics plot as an image and get the filename
        plot_filename = chatbot_trainer.plot_and_save_training_metrics(history, speaker)
        chatbot_trainer.logger.info(f"Training metrics plot saved as {plot_filename}")
