import os
import re
import numpy as np
from keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import logging
import pickle
import convokit
import time
import json
import pdb


def load_processed_dialogs(json_path):
    with open(json_path, "r") as f:
        grouped_dialogues = json.load(f)

    # Convert from JSON format back to Python dictionary format
    python_dict = {}
    for conversation_id, dialog_groups in grouped_dialogues.items():
        if conversation_id == "0":  # Handle misc dialogues
            python_dict[conversation_id] = [
                [(dialog["person1"], dialog["person2"]) for dialog in misc_dialog]
                for misc_dialog in dialog_groups
            ]
        else:  # Handle normal grouped dialogues
            python_dict[conversation_id] = [
                (dialog["person1"], dialog["person2"]) for dialog in dialog_groups
            ]

    return python_dict

processed_dialogs = load_processed_dialogs("processed_dialogs.json")

class BeamSearchHelper:
    def __init__(self, model, tokenizer, max_seq_length, encoder_filename, decoder_filename, top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.top_k = top_k
        self.encoder_filename = encoder_filename
        self.decoder_filename = decoder_filename
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
        logger.addHandler(file_handler)

        return logger

    def beam_search(self, input_seq, beam_width=3):
        # Load models
        encoder_model = load_model(self.encoder_filename)
        decoder_model = load_model(self.decoder_filename)

        # Encode the input sequence
        states_value = encoder_model.predict(input_seq)

        sequences = [[list(), 1.0, states_value]]

        # Walk over each step in sequence
        for _ in range(self.max_seq_length):
            all_candidates = list()
            for i in range(len(sequences)):
                seq, score, states = sequences[i]
                if len(seq) > 0 and seq[-1] == self.tokenizer.word_index.get('<end>'):
                    all_candidates.append(sequences[i])
                    continue

                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = self.tokenizer.word_index.get('<start>') if len(seq) == 0 else seq[-1]

                output_tokens, h, c = decoder_model.predict([target_seq] + states)
                states = [h, c]

                for j in range(len(output_tokens[0, -1, :])):
                    candidate = [seq + [j], score * -np.log(output_tokens[0, -1, j]), states]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beam_width]

        return sequences[0][0]

class BeamState:
    def __init__(self, sequence, score, state, logger):
        self.sequence = sequence
        self.score = score
        self.state = state
        self.logger = logger

    def __lt__(self, other):
        return self.score < other.score

    def log(self, message):
        self.logger.debug(message)



class MonitorEarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=3, mode='min', restore_best_weights=True, verbose=1):
        super(MonitorEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_weights = None
        self.best_epoch = None
        self.wait = 0
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.stopped_epoch_list = []  # List to track stopped epochs

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            if self.verbose > 0:
                print(f"Warning: Metric '{self.monitor}' is not available in logs.")
            return

        # Check for improvement based on mode
        if (self.mode == 'min' and current_value < self.best_value) or (self.mode == 'max' and current_value > self.best_value):
            self.best_value = current_value
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            self.wait = 0
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: {self.monitor} improved to {self.best_value:.4f}")
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: {self.monitor} did not improve. Patience: {self.wait}/{self.patience}")

            # Stop training if patience is exceeded
            if self.wait >= self.patience:
                self.stopped_epoch_list.append(epoch + 1)  # Record the stopped epoch
                if self.verbose > 0:
                    print(f"Stopping early at epoch {epoch + 1}. Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch + 1}")
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print(f"Restoring best model weights from epoch {self.best_epoch + 1}.")
                    self.model.set_weights(self.best_weights)


class ChatbotTrainer:
    def __init__(self):
        self.corpus = None
        self.all_vocab_size = 0
        self.model = None
        self.name = "Alan"
        self.model_filename = "Alan_model.keras"
        self. encoder_filename = "encoder.keras"
        self.decoder_filename = "decoder.keras"
        self.tokenizer_save_path = "chatBotTokenizer.pkl"
        self.tokenizer = None
        self.reverse_tokenizer = None
        self.early_patience = 11
        self.embedding_dim = 128
        self.max_seq_length = 64
        self.learning_rate = 0.00135
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.batch_size = 64
        self.epochs = 22
        self.vocabularyList = []
        self.troubleList = []
        self.max_vocab_size = None
        self.config = None
        self.max_vocabulary = 50000
        self.lstm_units = 512
        self.dropout = 0.3
        self.recurrent_dropout = 0.3
        self.test_size = 0.25
        self.logger = self.setup_logger()  # Initialize your logger here
        # Log Metrics...
        self.logger.info(f"""Metrics:\n
            Embedding/MaxSeqLength:({self.embedding_dim}, {self.max_seq_length})\n
            Batch Size: {self.batch_size}\n
            LSTM Units: {self.lstm_units}\n
            Epochs: {self.epochs}\n
            Dropout: ({self.dropout}, {self.recurrent_dropout})\n
            Test Split: {self.test_size}\n\n""")

        self.encoder_model = None
        self.encoder_inputs = None
        self.decoder_inputs = None
        self.decoder_outputs = None
        self.decoder_model = None

        if os.path.exists(self.tokenizer_save_path):
            with open(self.tokenizer_save_path, 'rb') as tokenizer_load_file:
                self.tokenizer = pickle.load(tokenizer_load_file)
                self.reverse_tokenizer = {index: word for word, index in self.tokenizer.word_index.items()}
                self.all_vocab_size = self.tokenizer.num_words
                for words, i in self.tokenizer.word_index.items():
                    if words not in self.vocabularyList:
                        self.vocabularyList.append(words)
                self.logger.info("Tokenizer loaded successfully.")
                print(f"Number of words in loaded tokenizer: {len(self.tokenizer.word_index)}")
                print(f"Number of words in the Vocab List: {len(self.vocabularyList)}")
        else:
            self.logger.warning("Tokenizer not found, making now...  ")
            self.tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-/.:;=?@[\\]^_`{|}~\t\n')

            # Save '<OOV>', '<start>', and '<end>' to word index
            self.tokenizer.num_words = 0
            self.vocabularyList = ['<start>', '<end>']
            for token in self.vocabularyList:
                if token not in self.tokenizer.word_index:
                    self.tokenizer.word_index[token] = self.tokenizer.num_words
                    self.all_vocab_size += 1
                    self.tokenizer.num_words += 1

            # Set Tokenizer Values:
            self.tokenizer.num_words = len(self.tokenizer.word_index)
            self.tokenizer.oov_token = "<oov>"

            self.logger.info(f"New Tokenizer Index's:  {self.tokenizer.word_index}")


        # Debug Line
        print(list(self.tokenizer.word_index.keys()))

        if os.path.exists(self.model_filename) and os.path.exists(self.encoder_filename) and os.path.exists(self.decoder_filename):
            self.model, self.encoder_model, self.decoder_model =self.load_model_file()

    def save_full_weights(self, encoder_path="encoder_weights.h5", decoder_path="decoder_weights.h5"):
        """
        Save the weights of the encoder and decoder to separate files.
        """
        if self.encoder_model is not None and self.decoder_model is not None:
            if os.path.exists(encoder_path):
                os.remove(encoder_path)
            if os.path.exists(decoder_path):
                os.remove(decoder_path)
            self.encoder_model.save_weights(encoder_path)
            self.decoder_model.save_weights(decoder_path)
            self.logger.info(f"Encoder weights saved at {encoder_path}.")
            self.logger.info(f"Decoder weights saved at {decoder_path}.")
        else:
            self.logger.warning(
                "Encoder or Decoder model does not exist. Ensure models are initialized before saving weights.")

    def load_full_weights(self, encoder_path="encoder_weights.h5", decoder_path="decoder_weights.h5"):
        """
        Load weights into the encoder and decoder models from separate files.
        """
        if self.encoder_model is not None and self.decoder_model is not None:
            self.encoder_model.load_weights(encoder_path)
            self.decoder_model.load_weights(decoder_path)
            self.logger.info(f"Encoder weights loaded from {encoder_path}.")
            self.logger.info(f"Decoder weights loaded from {decoder_path}.")
        else:
            self.logger.warning(
                "Encoder or Decoder model does not exist. Ensure models are initialized before loading weights.")

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
        plot_filename = os.path.join("D:\\ChatBotMetrics", f"{speaker}_training_metrics.png")
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

    def save_tokenizer(self, texts=None):
        if self.tokenizer:
            if texts:
                for token in texts:
                    if token not in self.tokenizer.word_index and self.tokenizer.num_words < self.max_vocabulary:
                        self.tokenizer.word_index[token] = self.tokenizer.num_words
                        self.all_vocab_size += 1
                        self.tokenizer.num_words += 1
                        # Debug Line
                        # print(f"Word: {token}\nIndex: {self.tokenizer.num_words}")
                        self.max_vocab_size = self.tokenizer.num_words

                self.tokenizer.fit_on_texts(texts)

            with open(self.tokenizer_save_path, 'wb') as tokenizer_save_file:
                pickle.dump(self.tokenizer, tokenizer_save_file)

            self.tokenizer.num_words = len(self.tokenizer.word_index)

        elif self.tokenizer == None:
            self.logger.warning("No tokenizer to save.")

    def save_embedding_weights(self, filepath="embedding_weights.npy"):
        """
        Save the weights of the embedding layer to a file.
        """
        if self.model is not None:
            embedding_layer = self.model.get_layer('embedding')

            # Extract the weights
            embedding_weights = embedding_layer.get_weights()[0]  # Weights are stored as a list, take the first element

            # Save weights to a file
            if os.path.exists(filepath):
                os.remove(filepath)

            np.save(filepath, embedding_weights)
            self.logger.info(f"Embedding weights saved successfully at {filepath}.")
        else:
            self.logger.warning("No model exists to extract embedding weights.")

    def load_embedding_weights(self, filepath="embedding_weights.npy"):
        """
        Load weights into the embedding layer from a file.
        """
        if self.model is not None:
            embedding_layer = self.model.get_layer('embedding')

            # Load weights from the file
            embedding_weights = np.load(filepath)

            # Ensure the weights shape matches the layer's expected shape
            if embedding_layer.input_dim == embedding_weights.shape[0] and embedding_layer.output_dim == \
                    embedding_weights.shape[1]:
                embedding_layer.set_weights([embedding_weights])
                self.logger.info(f"Embedding weights loaded successfully from {filepath}.")
            else:
                self.logger.error("Mismatch in embedding weights shape. Ensure the model and weights are compatible.")
        else:
            self.logger.warning("No model exists to load embedding weights into.")

    def clean_text(self, text):
        # Remove apostrophes from words containing letters and apostrophes
        text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"", text)  # Remove apostrophe within words

        # Remove any remaining non-alphanumeric characters (except spaces)
        text = re.sub(r"[^a-zA-Z0-9 ]", ' ', text)  # Keep letters, numbers, and spaces

        # Collapse extra spaces and strip leading/trailing spaces
        text = re.sub(r"\s+", " ", text).strip()

        for words in text.split(" "):
            if words not in self.vocabularyList:
                self.vocabularyList.append(words)

        return text

    # Training
    def preprocess_texts(self, input_texts, target_texts):
        input_texts = [self.clean_text(text) for text in input_texts]
        target_texts = [self.clean_text(text) for text in target_texts]
        self.save_tokenizer(self.vocabularyList)
        # Initialize lists to store processed inputs and targets
        input_texts = [f"<start> {texts} <end>" for texts in input_texts if input_texts and input_texts != "" and input_texts is not None]
        target_texts = [f"<start> {texts} <end>" for texts in target_texts if target_texts and target_texts != "" and target_texts is not None]

        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)

        input_sequences = pad_sequences(input_sequences, maxlen=self.max_seq_length, padding='post')
        target_sequences = pad_sequences(target_sequences, maxlen=self.max_seq_length, padding='post')

        return input_sequences, target_sequences

    # Prediction
    def preprocess_input(self, texts):
        # Assuming texts is a list of sentences
        preprocessed_input = ["<start>"]
        texts = self.clean_text(texts)

        preprocessed_text = texts.lower().split(" ")
        for words in preprocessed_text:
            preprocessed_input.append(words)

        preprocessed_input.append('<end>')

        print(list(preprocessed_input))
        preprocessed_input = self.tokenizer.texts_to_sequences(preprocessed_input)
        preprocessed_input = pad_sequences(preprocessed_input, maxlen=self.max_seq_length, padding='post')
        print(list(preprocessed_input))
        return preprocessed_input

    def preprocess_config(self, config):
        """
        Convert any non-serializable values (e.g., float32) to Python-native types.
        """
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = self.preprocess_config(value)
            elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                config[key] = float(value)
        return config

    def build_model(self):
        if not self.model:
            # Encoder
            self.encoder_inputs = Input(shape=(self.max_seq_length,))
            encoder_embedding = Embedding(
                input_dim=self.max_vocabulary,
                output_dim=self.embedding_dim,
                embeddings_regularizer=l2(0.01)
            )(self.encoder_inputs)
            encoder_lstm = LSTM(
                self.lstm_units,
                return_state=True,
                return_sequences=False,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout
            )
            _, state_h, state_c = encoder_lstm(encoder_embedding)
            encoder_states = [state_h, state_c]
            self.encoder_model = Model(self.encoder_inputs, encoder_states)

            # Decoder
            self.decoder_inputs = Input(shape=(None,), name='decoder_input')
            decoder_embedding = Embedding(
                input_dim=self.max_vocabulary,
                output_dim=self.embedding_dim
            )(self.decoder_inputs)
            decoder_lstm = LSTM(
                self.lstm_units,
                return_sequences=True,
                return_state=True,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l2(0.001)
            )
            decoder_state_input_h = Input(shape=(self.lstm_units,))
            decoder_state_input_c = Input(shape=(self.lstm_units,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_lstm_output, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_dense = Dense(self.max_vocabulary, activation='softmax')
            self.decoder_outputs = decoder_dense(decoder_lstm_output)
            self.decoder_model = Model([self.decoder_inputs] + decoder_states_inputs,
                                       [self.decoder_outputs] + decoder_states)

            # Combine encoder and decoder into the full model
            decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
            self.decoder_outputs = decoder_dense(decoder_lstm_output)
            self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
            self.model.compile(
                optimizer=self.optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return self.model, self.encoder_model, self.decoder_model

    def train_model(self, input_texts, target_texts, conversation_id, speaker):
        config_name = "model_config.json"
        if os.path.exists(config_name):
            with open(config_name, 'r') as fr:
                self.config = json.load(fr)
        self.logger.info(f"Training Model for ConversationID: {conversation_id}")

        # Load existing models if available
        if os.path.exists(self.model_filename) and os.path.exists(self.encoder_filename) and os.path.exists(
                self.decoder_filename):
            self.model, self.encoder_model, self.decoder_model = self.load_model_file()

        if self.corpus is None or self.tokenizer is None:
            raise ValueError("Corpus or tokenizer is not initialized.")

        # Preprocess the texts into sequences (Saves tokenizer)
        input_sequences, target_sequences = self.preprocess_texts(input_texts, target_texts)

        # Stats
        self.logger.info(f"Num Words: {self.tokenizer.num_words}")
        self.logger.info(f"Vocabulary Size: {len(self.tokenizer.word_index)}")
        self.logger.info(f"Length of Vocabulary List: {len(self.vocabularyList)}")

        # Build the model if it doesn't exist
        if not self.model and not self.encoder_model and not self.decoder_model:
            self.model, self.encoder_model, self.decoder_model = self.build_model()

        # Prepare training data
        encoder_input_data = input_sequences
        decoder_input_data = target_sequences[:, :-1]
        decoder_target_data = target_sequences[:, 1:]

        self.logger.info(f"Encoder Input Data Shape: {encoder_input_data.shape}")
        self.logger.info(f"Decoder Input Data Shape: {decoder_input_data.shape}")
        self.logger.info(f"Decoder Target Data Shape: {decoder_target_data.shape}")

        # Instantiate the callback
        early_stopping = MonitorEarlyStopping(
            monitor='val_loss',
            patience=self.early_patience,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )

        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

        # Train the model
        history = self.model.fit(
            [encoder_input_data, decoder_input_data],
            np.expand_dims(decoder_target_data, -1),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.test_size,
            callbacks=[early_stopping, lr_scheduler]
        )

        # Log any early stopping events
        if len(early_stopping.stopped_epoch_list) > 0:
            self.troubleList.append(speaker)

        # Reset stopped epoch list
        early_stopping.stopped_epoch_list = []

        # Evaluate the model on the training data
        test_loss, test_accuracy = self.model.evaluate(
            [encoder_input_data, decoder_input_data],
            np.expand_dims(decoder_target_data, -1),
            batch_size=self.batch_size
        )

        # Save training metrics as a plot
        plot_filename = self.plot_and_save_training_metrics(history, speaker)
        self.logger.info(f"Training metrics plot saved as {plot_filename}")
        self.logger.info(f"Test loss for Conversation {speaker}: {test_loss}")
        self.logger.info(f"Test accuracy for Conversation {speaker}: {test_accuracy}")
        self.logger.info(f"Model trained and saved successfully for speaker: {speaker}")

        # Compile the model before saving
        self.model.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Save the model after training
        self.save_model(self.model, self.encoder_model, self.decoder_model)

    def save_model(self, model, encoder_model, decoder_model):
        self.logger.info("Saving Model...")
        if model:
            self.encoder_model.save(self.encoder_filename)
            self.logger.info("Encoder saved.")
            self.decoder_model.save(self.decoder_filename)
            self.logger.info("Decoder saved.")
            self.model.save(self.model_filename)
            self.logger.info("Model saved.")
            self.save_full_weights()
            self.save_embedding_weights()

        else:
            self.logger.warning("No model to save.")

    def load_model_file(self):
        self.logger.info("Loading Model and Tokenizer...")

        model = load_model(self.model_filename)
        encoder_model = load_model(self.encoder_filename)
        decoder_model = load_model(self.decoder_filename)

        self.load_full_weights()
        self.load_embedding_weights()

        return model, encoder_model, decoder_model

    def predict_sequence(self, input_seq):
        # Ensure input_seq is properly padded and shaped
        input_seq = pad_sequences(input_seq, maxlen=self.max_seq_length, padding='post')
        encoder_model = load_model(self.encoder_filename)
        decoder_model = load_model(self.decoder_filename)
        
        states_value = encoder_model.predict(input_seq)
        
        # Initialize the decoder input with the start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.tokenizer.word_index.get('<start>', 1)

        stop_condition = False
        decoded_sentence = []

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.tokenizer.index_word.get(sampled_token_index, '<oov>')
            
            decoded_sentence.append(sampled_token)

            if sampled_token == '<end>' or len(decoded_sentence) > self.max_seq_length:
                stop_condition = True
            
            # Update target sequence and states
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

        return ' '.join(decoded_sentence)

    def generate_response_with_beam_search(self, user_input, beam_width=3):
        # Preprocess user input
        user_input = self.preprocess_text(user_input)
        user_input_seq = self.tokenizer.texts_to_sequences(user_input)
        user_input_seq = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post')

        reverse_target_char_index = dict(map(reversed, self.tokenizer.word_index.items()))

        beamHelper = BeamSearchHelper(self.model, self.tokenizer, self.max_seq_length)

        # Perform beam search
        response_sequences = beamHelper.beam_search(user_input_seq, beam_width=beam_width)

        # Convert sequences to texts
        response_texts = [' '.join([reverse_target_char_index[token] for token in seq if token in reverse_target_char_index]) for seq in response_sequences]

        response_string = " ".join(response_texts)
        return response_string

    def generate_response(self, input_seq):
        """
        Generates a response from the chatbot for the given input sequence.
        """
        try:
            # Clean and tokenize input text
            input_seqs = self.preprocess_input(input_seq)

            # Pad the input sequence to the required length
            input_seq = pad_sequences(input_seqs, maxlen=self.max_seq_length, padding='post')

            # Encode the input sequence using the encoder model
            encoder_states = self.encoder_model.predict(input_seq)
            state_h, state_c = encoder_states

            # Initialize the decoder input with the <start> token
            start_token_index = self.tokenizer.word_index.get('<start>', 1)
            target_seq = np.zeros((len(input_seqs), 1))  # Batch size x 1
            target_seq[:, 0] = start_token_index

            # Debugging before passing to the decoder
            print(f"Initial Target Seq Shape: {target_seq.shape}, state_h Shape: {state_h.shape}, state_c Shape: {state_c.shape}")

            # Special Tokens (Excluded from Response)
            special_tokens = ['<oov>', '<start>', '<end>']

            # Decode the sequence + Make sure the reverse tokenizer is instantiated properly
            self.reverse_tokenizer = {v: k for k, v in self.tokenizer.word_index.items()}
            decoded_sentence = []
            for _ in range(self.max_seq_length):
                # Decoder expects target_seq and states (state_h, state_c)
                output_tokens, state_h, state_c = self.decoder_model.predict([target_seq, state_h, state_c])

                # Get the token with the highest probability
                print(output_tokens.shape)
                print(f"Output Token Probabilities: {output_tokens[0, -1, :]}")
                predicted_token_index = np.argmax(output_tokens[0, -1, :])
                # Ensure reverse_tokenizer is built correctly
                predicted_word = self.reverse_tokenizer.get(predicted_token_index, "<oov>")

                if predicted_word == "<end>":  # Stop decoding if <end> token is encountered
                    break

                if predicted_word not in special_tokens:
                    # Append the word to the decoded sentence
                    decoded_sentence.append(predicted_word)

                # Update target sequence with the predicted token index for the next iteration
                target_seq[0, 0] = predicted_token_index

            return " ".join(decoded_sentence).strip()  # Return the decoded sentence

        except Exception as e:
            self.logger.error(f"Error in generate_response: {str(e)}")
            return "I'm sorry, I encountered an error while generating a response."
