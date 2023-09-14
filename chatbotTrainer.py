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


class ChatbotTrainer:
    def __init__(self):
        self.corpus = None
        self.max_vocab_size = max_vocab_size = 50000
        self.model = None
        self.model_filename = "chatbot_model.h5"
        self.tokenizer_save_path = "chatBotTokenizer.pkl"
        self.tokenizer = None
        self.logger = self.setup_logger()  # Initialize your logger here
        self.embedding_dim = 100  # Define the embedding dimension here
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
        # Add '<start>' to beginning and '<end>'
        text = f"<start> {text} <end>"
        # Remove double quotes from the text
        cleaned_text = text.replace('"', '')

        # Remove multiple spaces
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text.lower()


    def save_tokenizer(self, texts=None):
        if texts:
            # Fit the tokenizer on the provided texts
            self.tokenizer.fit_on_texts(texts)

        if self.tokenizer:
            with open(self.tokenizer_save_path, 'wb') as tokenizer_save_file:
                pickle.dump(self.tokenizer, tokenizer_save_file)
        else:
            self.logger.warning("No tokenizer to save.")


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
        encoder_lstm, state_h, state_c = LSTM(units=lstm_units, return_state=True, dropout=0.10)(encoder_embedding)  # Added dropout
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(max_seq_length,))
        decoder_embedding = Embedding(input_dim=vocab_size, output_dim=self.embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(units=lstm_units, return_sequences=True, return_state=True, dropout=0.10)  # Added dropout
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


    def train_model(self, input_texts, target_texts, conversation_id):
        self.logger.info("Training Model...")

        if self.corpus is None or self.tokenizer is None:
            raise ValueError("Corpus or tokenizer is not initialized.")

        # Tokenize input and target texts
        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)

        max_seq_length = self.max_seq_length  # Specify your maximum sequence length
        padded_input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
        padded_target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

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

        # Save training metrics plot as an image and get the filename
        plot_filename = self.plot_and_save_training_metrics(history, conversation_id)
        self.logger.info(f"Training metrics plot saved as {plot_filename}")

        return history  # Return the history object


    def save_model(self):
        self.logger.info("Saving Model...")
        if self.model:
            self.model.save(self.model_filename)
        else:
            self.logger.warning("No model to save.")

    def load_model(self):
        self.logger.info("Loading Model and Tokenizer...")
        if os.path.exists(self.model_filename):
            # Load both the model and tokenizer
            if os.path.exists(self.tokenizer_save_path):
                with open(self.tokenizer_save_path, 'rb') as tokenizer_load_file:
                    self.tokenizer = pickle.load(tokenizer_load_file)
                    self.tokenizer.num_words = self.max_vocab_size
                    self.model = tf.keras.models.load_model(self.model_filename)
                    self.logger.info("Model and tokenizer loaded successfully.")
            else:
                print("Tokenizer not found, making now...  ")
                self.tokenizer = Tokenizer(oov_token="<OOV>", num_words=self.max_vocab_size)  # Initialize the Tokenizer
                self.tokenizer.num_words = self.max_vocab_size

        else:
            self.logger.warning("No saved model found... Making now...  ")
            # Build the model (if not already built)
            if self.model is None:
                self.build_model()


    def beam_search(self, input_seq, beam_width=3, max_length=50):
        start_token = self.tokenizer.word_index['<start> ']
        end_token = self.tokenizer.word_index[' <end>']

        # Initialize beam search with a single hypothesis
        initial_state = self.model.layers[2].initialize_states(batch_size=self.batch_size)
        initial_state = [initial_state, initial_state]
        initial_beam = BeamState(score=0.0, sequence=[start_token], state=initial_state)

        beam_states = [initial_beam]

        # Perform beam search
        for _ in range(max_length):
            new_beam_states = []
            for state in beam_states:
                if state.sequence[-1] == end_token:
                    # If the hypothesis ends, add it to the final hypotheses
                    new_beam_states.append(state)
                else:
                    # Generate next token probabilities and states
                    decoder_input = np.array([state.sequence[-1]])
                    decoder_state = state.state

                    decoder_output, decoder_state = self.model.layers[2](decoder_input, initial_state=decoder_state)
                    token_probs = decoder_output[0, 0]

                    # Get the top beam_width tokens
                    top_tokens = np.argsort(token_probs)[-beam_width:]

                    for token in top_tokens:
                        new_seq = state.sequence + [token]
                        new_score = state.score - np.log(token_probs[token])
                        new_state = decoder_state

                        new_beam_states.append(BeamState(score=new_score, sequence=new_seq, state=new_state))

            # Select top beam_width hypotheses
            new_beam_states.sort(key=lambda x: x.score)
            beam_states = new_beam_states[:beam_width]

        # Get the hypothesis with the highest score
        best_hypothesis = max(beam_states, key=lambda x: x.score)
        return best_hypothesis.sequence[1:]  # Exclude the start token


    def generate_response(self, user_input, beam_width=3):
        user_input = self.preprocess_text(user_input)
        user_input_seq = self.tokenizer.texts_to_sequences([user_input])
        user_input_seq = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post')

        response_sequence = self.beam_search(user_input_seq, beam_width=beam_width)
        response_text = self.tokenizer.sequences_to_texts([response_sequence])[0]

        return response_text


class BeamState:
    def __init__(self, score, sequence, state):
        self.score = score
        self.sequence = sequence
        self.state = state

