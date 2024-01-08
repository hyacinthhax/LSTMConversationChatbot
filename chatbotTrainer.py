import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import pickle
import convokit
from processed_dialogs import dialog_data  # Import the dialog_data dictionary
from playsound import playsound


class BeamSearchHelper:
    def __init__(self, model, tokenizer, max_seq_length, top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.top_k = top_k
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger("ChatbotBeamSearch")
        logger.setLevel(logging.DEBUG)

        # Create console handler and set level to INFO for progress reports
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Create a file handler and set level to DEBUG for progress reports and ERROR for error notifications
        file_handler = logging.FileHandler("chatbotBeam.log")
        file_handler.setLevel(logging.DEBUG)  # Set level to DEBUG to capture progress reports
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger


    def beam_search(self, input_seqs, beam_width=3, max_length=100):
        # Find the correct index of the LSTM layer
        lstm_layer_index = None
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, LSTM):
                self.logger.info(f"LSTM layer found at index {i}: {layer}")
                lstm_layer_index = i

        if lstm_layer_index is not None:
            lstm_layer = self.model.layers[lstm_layer_index]

        # Initialize beam search with a single hypothesis for each input sequence
        initial_states = [lstm_layer.get_initial_state(inputs=tf.constant([seq])) for seq in input_seqs]
        initial_beams = [BeamState(state=state, score=0.0, sequence=[start_token], logger=self.logger) for state in initial_states]

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

                    # Get the top-k tokens for each input sequence
                    top_k_tokens = np.argsort(token_probs, axis=-1)[:, -self.top_k:]

                    for seq_idx in range(len(input_seqs)):
                        for token in top_k_tokens[seq_idx]:
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


class BeamState:
    def __init__(self, score, sequence, state, logger):
        if not isinstance(score, (float, int)):
            logger.warning(f"Warning: Invalid score type: {type(score)}")
        if not isinstance(sequence, list):
            logger.warning(f"Warning: Invalid sequence type: {type(sequence)}")
        if not isinstance(state, np.ndarray):  # Adjust the type accordingly
            logger.warning(f"Warning: Invalid state type: {type(state)}")

        self.score = score
        self.sequence = sequence
        self.state = state



class ChatbotTrainer:
    def __init__(self):
        self.corpus = None
        self.all_vocab_size = 0
        self.model = None
        self.model_filename = "chatbot_model.h5"
        self.tokenizer_save_path = "chatBotTokenizer.pkl"
        self.tokenizer = None
        self.logger = self.setup_logger()  # Initialize your logger here
        self.embedding_dim = 100  # Define the embedding dimension here
        self.max_seq_length = 100  # Replace with your desired sequence length
        self.learning_rate = 0.003
        self.batch_size = 128
        self.epochs = 3
        self.lstm_units = 256
        self.vocabularyList = ['<start>', '<end>', '<OOV>']
        self.speakerList = []
        self.encoder_model = None
        self.encoder_inputs = None
        self.decoder_inputs = None
        self.decoder_outputs = None
        self.decoder_model = None

        # Import Speakers
        with open('trained_speakers.txt', 'r') as file:
            self.speakerList = file.read().splitlines()

        if os.path.exists(self.tokenizer_save_path):
            with open(self.tokenizer_save_path, 'rb') as tokenizer_load_file:
                self.tokenizer = pickle.load(tokenizer_load_file)
                self.all_vocab_size = self.tokenizer.num_words
                self.logger.info("Model and tokenizer loaded successfully.")
        else:
            self.logger.warning("Tokenizer not found, making now...  ")
            self.tokenizer = Tokenizer(num_words=0, oov_token="<OOV>")  # Initialize the Tokenizer

            # Save '<OOV>', '<start>', and '<end>' to word index
            for token in self.vocabularyList:
                if token not in self.tokenizer.word_index:
                    self.tokenizer.word_index[token] = self.tokenizer.num_words
                    self.all_vocab_size += 1
                    self.tokenizer.num_words += 1

            self.logger.info(f"New Tokenizer Index's:  {self.tokenizer.word_index}")
            self.save_tokenizer()

        self.load_model_file()

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
        plot_filename = os.path.join("C:\\Users\\admin\\Desktop\\ChatBotMetrics", f"{speaker}_training_metrics.png")
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


    def preprocess_text(self, text):
        # Inputs a String of Text Converts as that string whole
        blacklist = ['"', '<', '>', "'"]
        cleaned_text = []
        for words in text:
            if words not in blacklist:
                cleaned_text.append(words)

        cleaned_text = ''.join(cleaned_text)
        # Remove multiple spaces
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        cleaned_text = re.sub(r"[^A-Za-z0-9 ]+", "", cleaned_text)
        for words in cleaned_text.split():
            self.vocabularyList.append(words)
        # Add '<start>' to beginning and '<end>'
        cleaned_text = f"<start> {cleaned_text} <end>"
        return cleaned_text.lower()

    def save_tokenizer(self, texts=None):
        if self.tokenizer:
            if texts:
                self.tokenizer.fit_on_texts(texts)

            self.tokenizer.num_words = self.all_vocab_size

            with open(self.tokenizer_save_path, 'wb') as tokenizer_save_file:
                pickle.dump(self.tokenizer, tokenizer_save_file)

        elif self.tokenizer == None:
            self.logger.warning("No tokenizer to save.")

    def build_model(self):
        self.logger.info("Building model...")
        lstm_units = self.lstm_units
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")

        max_seq_length = self.max_seq_length
        learning_rate = self.learning_rate

        # Encoder
        self.encoder_inputs = Input(shape=(max_seq_length,))
        encoder_embedding = Embedding(input_dim=self.all_vocab_size, output_dim=self.embedding_dim)(self.encoder_inputs)
        encoder_lstm, state_h, state_c = LSTM(units=lstm_units, return_state=True, dropout=0.2, recurrent_dropout=0.1)(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        self.decoder_inputs = Input(shape=(max_seq_length,))
        decoder_embedding = Embedding(input_dim=self.all_vocab_size, output_dim=self.embedding_dim)(self.decoder_inputs)
        decoder_lstm = LSTM(units=lstm_units, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.1)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        decoder_dense = Dense(units=self.all_vocab_size, activation='softmax')
        self.decoder_outputs = decoder_dense(decoder_outputs)

        # Create the model
        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

        # Compilation
        self.model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Inference
        self.encoder_model = Model(self.encoder_inputs, encoder_states)

        self.decoder_state_input_h = Input(shape=(lstm_units,))
        self.decoder_state_input_c = Input(shape=(lstm_units,))
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=self.decoder_states_inputs)
        decoder_states = [state_h, state_c]

        self.decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_model = Model([self.decoder_inputs] + self.decoder_states_inputs, [self.decoder_outputs] + decoder_states)


    def train_model(self, input_texts, target_texts, conversation_id, speaker):
        # Save the speakerList
        with open('trained_speakers.txt', 'a') as file:
            file.write(f'{speaker}\n')

        self.logger.info(f"Training Model... \nConversationID:  {conversation_id}")

        if self.corpus is None or self.tokenizer is None:
            raise ValueError("Corpus or tokenizer is not initialized.")

        # save_tokenizer and update num_words before training
        for lines in input_texts:
            self.save_tokenizer(lines.split())
        self.logger.info(f"Word Counts:  {self.tokenizer.word_counts}")

        # Preprocess the training data using the tokenizer
        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        padded_input_sequences = pad_sequences(input_sequences, maxlen=self.max_seq_length, padding='post')
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)
        padded_target_sequences = pad_sequences(target_sequences, maxlen=self.max_seq_length, padding='post')

        # Split this speaker's data into training and test sets
        train_input, test_input, train_target, test_target = train_test_split(padded_input_sequences, padded_target_sequences, test_size=0.2, random_state=42)

        # Train the model
        history = self.model.fit([train_input, train_target], train_target, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

        # Evaluate the model on the test data
        test_loss, test_accuracy = self.model.evaluate([test_input, test_target], test_target, batch_size=self.batch_size)

        # Save Best progress to new h5 to avoid losing data
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)

        # Save the model
        self.save_model()

        # Log training metrics
        self.logger.info("Training metrics:")
        self.logger.info("Model trained successfully.")

        # Save training metrics plot as an image and get the filename
        plot_filename = self.plot_and_save_training_metrics(history, conversation_id)
        self.logger.info(f"Training metrics plot saved as {plot_filename}")
        self.logger.info(f"Test loss for Conversation {speaker}: {test_loss}")
        self.logger.info(f"Test accuracy for Conversation {speaker}: {test_accuracy}")


    def save_model(self):
        self.logger.info("Saving Model...")
        if self.model:
            self.model.save(self.model_filename)
        else:
            self.logger.warning("No model to save.")

    def load_model_file(self):
        self.logger.info("Loading Model and Tokenizer...")
        if os.path.exists(self.model_filename):
            # Load both the model and tokenizer using TensorFlow's load_model method
            self.model = tf.keras.models.load_model(self.model_filename)
            self.logger.info("Model and tokenizer loaded successfully.")

        elif not os.path.exists(self.model_filename):
            self.logger.warning("No saved model found... Making now...  ")
            # Build the model (if not already built)
            if self.model is None:
                self.build_model()


    def generate_response_with_beam_search(self, user_input, beam_width=3, batch_size=None):
        # Preprocess user input
        user_input = self.preprocess_text([user_input])[0]
        user_input_seq = self.tokenizer.texts_to_sequences([user_input])
        user_input_seq = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post')

        # Initialize target_seq with the index of <start> token
        start_token_index = self.tokenizer.word_index['<start>']
        end_token_index = self.tokenizer.word_index['<end>']
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = start_token_index

        reverse_target_char_index = dict(map(reversed, self.tokenizer.word_index.items()))

        beamHelper = BeamSearchHelper(self.model, self.tokenizer, self.max_seq_length)

        # Perform beam search
        response_sequences = beamHelper.beam_search(user_input_seq, beam_width=beam_width)

        token_index = ["<start>", "<end>", "<OOV>"]

        # Convert sequences to texts (MANUALLY)
        response_texts = []
        for seq in response_sequences:
             # Convert sequence to text using reverse_target_char_index
            response_text = ' '.join([reverse_target_char_index[token] for token in seq])
            response_texts.append(response_text)

        response_string = ""
        for response in response_texts:
            if response not in token_index:
                response_string = response_string + " " + response
                
        return response_string   # Or Uncomment below

        # Convert sequences to texts
        # response_texts = [self.tokenizer.sequences_to_texts([seq])[0] for seq in response_sequences]

        # response_string = ""
        # for response in response_texts:
            # if response not in token_index:
            # response_string = response_string + " " + response