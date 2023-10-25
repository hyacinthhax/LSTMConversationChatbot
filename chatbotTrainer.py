import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
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

        # Initialize the corpus
        corpus_path = "C:\\Users\\admin\\Desktop\\movie-corpus"
        if os.path.exists(self.tokenizer_save_path):
            with open(self.tokenizer_save_path, 'rb') as tokenizer_load_file:
                self.tokenizer = pickle.load(tokenizer_load_file)
                self.tokenizer.num_words = self.max_vocab_size
                self.logger.info("Model and tokenizer loaded successfully.")
                self.load_corpus(corpus_path)
        else:
            print("Tokenizer not found, making now...  ")
            self.tokenizer = Tokenizer(oov_token="<OOV>", num_words=self.max_vocab_size)  # Initialize the Tokenizer
            self.tokenizer.num_words = self.max_vocab_size
            self.load_corpus(corpus_path)


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
        cleaned_text = []
        for words in text:
            # Remove double quotes from the text
            words = words.replace('"', '')
            words = words.replace('<', '')
            words = words.replace('>', '')
            cleaned_text.append(words)

        cleaned_text = ''.join(cleaned_text)
        # Remove multiple spaces
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        cleaned_text = re.sub(r"[^A-Za-z0-9 ]+", "", cleaned_text)
        # Add '<start>' to beginning and '<end>'
        cleaned_text = f"<start> {cleaned_text} <end>"
        # print(cleaned_text)       # Debug Line for human verification
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
        encoder_lstm, state_h, state_c = LSTM(units=lstm_units, return_state=True, dropout=0.07)(encoder_embedding)  # Added dropout
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(max_seq_length,))
        decoder_embedding = Embedding(input_dim=vocab_size, output_dim=self.embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(units=lstm_units, return_sequences=True, return_state=True, dropout=0.07)  # Added dropout
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

        # Compile the model (no need to specify learning_rate here, it's done in build_model)
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy
                           metrics=['accuracy'])

        # Train the model
        history = self.model.fit(
            [input_texts, target_texts],
            target_texts,
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
                
                # Add "<start>" token to the word index if it doesn't already exist
                if '<start>' not in self.tokenizer.word_index:
                    self.tokenizer.word_index['<start>'] = self.tokenizer.num_words + 1
                    self.tokenizer.num_words += 1

                # Add "<end>" token to the word index if it doesn't already exist
                if '<end>' not in self.tokenizer.word_index:
                    self.tokenizer.word_index['<end>'] = self.tokenizer.num_words + 1
                    self.tokenizer.num_words += 1
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


    def beam_search(self, input_seqs, beam_width=3, max_length=100):
        start_token = self.tokenizer.word_index['<start>']
        end_token = self.tokenizer.word_index['<end>']
        
        # Find the correct index of the LSTM layer
        lstm_layer_index = None
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, LSTM):
                print(f"LSTM layer found at index {i}: {layer}")
                lstm_layer_index = i

        if lstm_layer_index is not None:
            lstm_layer = self.model.layers[lstm_layer_index]

        # Initialize beam search with a single hypothesis for each input sequence
        initial_states = [lstm_layer.get_initial_state(inputs=tf.constant([seq])) for seq in input_seqs]
        initial_beams = [BeamState(score=0.0, sequence=[start_token], state=state) for state in initial_states]

        beam_states = initial_beams

        # Perform beam search
        for _ in range(max_length):
            new_beam_states = []
            for state in beam_states:
                if state.sequence[-1] == end_token:
                    # If the hypothesis ends, add it to the final hypotheses
                    new_beam_states.append(state)
                else:
                    # Generate next token probabilities and states for all input sequences
                    decoder_input = tf.constant([[[state.sequence[-1]]] * len(input_seqs)])  # Repeat for all sequences
                    decoder_state = state.state

                    decoder_output, decoder_state, _ = lstm_layer(decoder_input, initial_state=decoder_state)
                    token_probs = decoder_output[:, 0, :]  # Slice for all sequences

                    # Get the top beam_width tokens for each input sequence
                    top_tokens = np.argsort(token_probs, axis=-1)[:, -beam_width:]

                    for seq_idx in range(len(input_seqs)):
                        for token in top_tokens[seq_idx]:
                            new_seq = state.sequence + [token]
                            new_score = state.score - np.log(token_probs[seq_idx, token])
                            new_state = decoder_state

                            new_beam_states.append(BeamState(score=new_score, sequence=new_seq, state=new_state))

            # Select top beam_width hypotheses for each input sequence
            new_beam_states = np.array(new_beam_states)
            best_indices = np.argsort(new_beam_states[:, 0])[:beam_width]

            beam_states = new_beam_states[best_indices]

        # Get the hypotheses with the highest scores for each input sequence
        best_hypotheses = [max(initial_beams, key=lambda x: x.score) for initial_beams in beam_states]
        return [hypothesis.sequence[1:] for hypothesis in best_hypotheses]  # Exclude the start token


    def generate_response(self, user_input, beam_width=3, batch_size=None):
        user_input = self.preprocess_text(user_input)
        user_input_seq = self.tokenizer.texts_to_sequences([user_input])
        user_input_seq = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post')
        
        if batch_size is None:
            response_sequence = self.beam_search(user_input_seq, beam_width=beam_width)
            response_text = self.tokenizer.sequences_to_texts([response_sequence])[0]
            return response_text
        else:
            # Split input into batches and generate responses batch by batch
            num_batches = len(user_input_seq) // batch_size
            responses = []

            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size
                batch_input = user_input_seq[batch_start:batch_end]

                batch_responses = self.beam_search(batch_input, beam_width=beam_width)
                for response_seq in batch_responses:
                    response_text = self.tokenizer.sequences_to_texts([response_seq])[0]
                    responses.append(response_text)

            return responses



class BeamState:
    def __init__(self, score, sequence, state):
        self.score = score
        self.sequence = sequence
        self.state = state

