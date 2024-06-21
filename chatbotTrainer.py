import os
import re
import numpy as np
from itertools import chain
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import pickle
import convokit
from processed_dialogs import dialog_data  # Import the dialog_data dictionary
import time
import pdb


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
        self.embedding_dim =  128 # Define the embedding dimension here HAS TO BE SAME AS MAX_SEQ_LENGTH and Replace with your desired sequence length(Max words in response)
        self.max_seq_length = 128
        self.learning_rate = 0.00222
        self.batch_size = 64
        self.epochs = 7
        self.vocabularyList = []
        self.max_vocab_size = None
        self.max_vocabulary = 30000
        self.lstm_units = 1024
        self.dropout = 0.3
        self.recurrent_dropout = 0.3
        self.validation_split = 0.2
        self.test_size = 0.1
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
                self.all_vocab_size = self.tokenizer.num_words
                for words, i in self.tokenizer.word_index.items():
                    if words not in self.vocabularyList:
                        self.vocabularyList.append(words)
                self.logger.info("Tokenizer loaded successfully.")
                print(f"Number of words in loaded tokenizer: {len(self.tokenizer.word_index)}")
                print(f"Number of words in the Vocab List: {len(self.vocabularyList)}")
        else:
            self.logger.warning("Tokenizer not found, making now...  ")
            self.tokenizer = Tokenizer(num_words=None)  # Initialize the Tokenizer

            # Save '<OOV>', '<start>', and '<end>' to word index
            self.tokenizer.num_words = 0
            self.vocabularyList = ['<PAD>', '<start>', '<end>', '<OOV>']
            for token in self.vocabularyList:
                if token not in self.tokenizer.word_index:
                    self.tokenizer.word_index[token] = self.tokenizer.num_words
                    self.all_vocab_size += 1
                    self.tokenizer.num_words += 1

            # Set Tokenizer Values:
            self.tokenizer.num_words = len(self.tokenizer.word_index)
            self.tokenizer.oov_token = "<OOV>"

            self.logger.info(f"New Tokenizer Index's:  {self.tokenizer.word_index}")
            self.save_tokenizer(self.vocabularyList)

        self.load_model_file(self.model_filename)

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
        plot_filename = os.path.join("E:\\ChatBotMetrics", f"{speaker}_training_metrics.png")
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
                    if token not in self.tokenizer.word_index:
                        self.all_vocab_size += 1
                        self.tokenizer.num_words += 1
                        self.tokenizer.word_index[token] = self.tokenizer.num_words
                        # Debug Line
                        # print(f"Word: {token}\nIndex: {self.tokenizer.num_words}")
                        self.max_vocab_size = self.tokenizer.num_words

                self.tokenizer.fit_on_texts(texts)

            with open(self.tokenizer_save_path, 'wb') as tokenizer_save_file:
                pickle.dump(self.tokenizer, tokenizer_save_file)

            self.tokenizer.num_words = len(self.tokenizer.word_index)

        elif self.tokenizer == None:
            self.logger.warning("No tokenizer to save.")
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
        return text

    # Training
    def preprocess_texts(self, input_texts, target_texts):
        input_texts = [self.clean_text(text) for text in input_texts]
        target_texts = [self.clean_text(text) for text in target_texts]
        # Initialize lists to store processed inputs and targets
        input_texts = [f"<start> {texts} <end>" for texts in input_texts]
        target_texts = [f"<start> {texts} <end>" for texts in target_texts]

        for text in input_texts:
            for words in text.split(" "):
                if words not in self.vocabularyList and words not in self.tokenizer.word_index.keys():
                    self.vocabularyList.append(words)

        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)

        input_sequences = pad_sequences(input_sequences, maxlen=self.max_seq_length, padding='post', truncating='post')
        target_sequences = pad_sequences(target_sequences, maxlen=self.max_seq_length, padding='post', truncating='post')

        return input_sequences, target_sequences

    # Prediction
    def preprocess_text(self, texts):
        # Assuming texts is a list of sentences
        preprocessed_texts = []
        for text in texts:
            # Example preprocessing: lowercase and split
            preprocessed_text = text.lower().split()
            preprocessed_texts.append(preprocessed_text)
        
        return preprocessed_texts
    
    def train_model(self, input_texts, target_texts, conversation_id, speaker):
        self.logger.info(f"Training Model for ConversationID: {conversation_id}")

        if self.corpus is None or self.tokenizer is None:
            raise ValueError("Corpus or tokenizer is not initialized.")

        input_sequences, target_sequences = self.preprocess_texts(input_texts, target_texts)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Save the tokenizer from VocabList
        self.save_tokenizer(self.vocabularyList)
        self.logger.info(f"Num Words:  {self.tokenizer.num_words}")
        self.logger.info(f"All Index:  {len(self.tokenizer.word_index)}")
        self.logger.info(f"Length VocabList:  {len(self.vocabularyList)}")

        # Debug Line
        # pdb.set_trace()

        if self.model is None:
            self.encoder_inputs = Input(shape=(self.max_seq_length,))
            encoder_embedding = Embedding(input_dim=self.max_vocabulary, output_dim=self.embedding_dim, embeddings_regularizer=l2(0.01))(self.encoder_inputs)
            encoder_lstm = LSTM(self.lstm_units, return_state=True, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)
            _, state_h, state_c = encoder_lstm(encoder_embedding)
            encoder_states = [state_h, state_c]
            self.encoder_model = Model(self.encoder_inputs, encoder_states)
            self.encoder_model.save(self.encoder_filename)

            self.decoder_inputs = Input(shape=(None,), name='decoder_input')
            decoder_embedding = Embedding(input_dim=self.max_vocabulary, output_dim=self.embedding_dim)(self.decoder_inputs)
            decoder_lstm = LSTM(self.lstm_units, return_sequences=True, return_state=True, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, kernel_regularizer=l2(0.01))
            decoder_state_input_h = Input(shape=(self.lstm_units,))
            decoder_state_input_c = Input(shape=(self.lstm_units,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_lstm_output, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_dense = Dense(self.max_vocabulary, activation='softmax')
            self.decoder_outputs = decoder_dense(decoder_lstm_output)
            self.decoder_model = Model([self.decoder_inputs] + decoder_states_inputs, [self.decoder_outputs] + decoder_states)
            self.decoder_model.save(self.decoder_filename)

            decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
            self.decoder_outputs = decoder_dense(decoder_lstm_output)
            self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        encoder_input_data, decoder_input_data = input_sequences, target_sequences[:, :-1]
        decoder_target_data = target_sequences[:, 1:]

        self.logger.info(f"Encoder Input Data Shape: {encoder_input_data.shape}")
        self.logger.info(f"Decoder Input Data Shape: {decoder_input_data.shape}")
        self.logger.info(f"Decoder Target Data Shape: {decoder_target_data.shape}")

        history = self.model.fit(
            [encoder_input_data, decoder_input_data],
            np.expand_dims(decoder_target_data, -1),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.test_size,
            callbacks=[early_stopping]
        )

        self.save_model(self.model, self.encoder_model, self.decoder_model)

        # Evaluate the model on the test data
        test_loss, test_accuracy = self.model.evaluate([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1), batch_size=self.batch_size)

        # Save training metrics plot as an image and get the filename
        plot_filename = self.plot_and_save_training_metrics(history, speaker)
        self.logger.info(f"Training metrics plot saved as {plot_filename}")
        self.logger.info(f"Test loss for Conversation {speaker}: {test_loss}")
        self.logger.info(f"Test accuracy for Conversation {speaker}: {test_accuracy}")
        self.logger.info(f"Model trained and saved successfully for speaker: {speaker}")
        time.sleep(30)

    def save_model(self, model, encoder_model, decoder_model):
        self.logger.info("Saving Model...")
        if self.model:
            model.save(self.model_filename)
            encoder_model.save(self.encoder_filename)
            decoder_model.save(self.decoder_filename)
            self.logger.info("Saved Model with Encoder and Decoder")
        else:
            self.logger.warning("No model to save.")

    def load_model_file(self, model_filename):
        self.logger.info("Loading Model and Tokenizer...")
        if os.path.exists(model_filename) and os.path.exists(self.encoder_filename) and os.path.exists(self.decoder_filename):
            # Load both the model and tokenizer using TensorFlow's load_model method
            self.model = load_model(model_filename)
            self.encoder_model = load_model(self.encoder_filename)
            self.decoder_model = load_model(self.decoder_filename)
            self.logger.info("Model and Encoder/Decoder loaded successfully.")

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
            sampled_token = self.tokenizer.index_word.get(sampled_token_index, '<OOV>')
            
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
        user_input = self.preprocess_text([user_input]).split(" ")
        user_input_seq = self.tokenizer.texts_to_sequences([user_input])
        user_input_seq = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post')

        reverse_target_char_index = dict(map(reversed, self.tokenizer.word_index.items()))

        beamHelper = BeamSearchHelper(self.model, self.tokenizer, self.max_seq_length)

        # Perform beam search
        response_sequences = beamHelper.beam_search(user_input_seq, beam_width=beam_width)

        # Convert sequences to texts
        response_texts = [' '.join([reverse_target_char_index[token] for token in seq if token in reverse_target_char_index]) for seq in response_sequences]

        response_string = " ".join(response_texts)
        return response_string


    def generate_response(self, user_input):
        start_token_id = self.tokenizer.word_index.get('<start>', 1)
        end_token_id = self.tokenizer.word_index.get('<end>', 2)

        # Preprocess user input
        user_input = self.preprocess_text([user_input])
        user_input_seq = self.tokenizer.texts_to_sequences([user_input])
        user_input_seq = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post')

        # Encode the input sequence
        encoder_model = load_model(self.encoder_filename)
        decoder_model = load_model(self.decoder_filename)
        states_value = encoder_model.predict(user_input_seq)

        # Initialize the decoder input with a start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = start_token_id

        stop_condition = False
        decoded_sentence = []
        
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.tokenizer.index_word.get(sampled_token_index, '<OOV>')

            decoded_sentence.append(sampled_token)

            # Exit condition: either hit max length or find stop token
            if sampled_token == end_token_id or len(decoded_sentence) > self.max_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        response_string = ' '.join(decoded_sentence)

        return response_string
