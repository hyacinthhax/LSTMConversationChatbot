import os
import re
import numpy as np
from itertools import chain
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import pickle
import convokit
from processed_dialogs import dialog_data  # Import the dialog_data dictionary
from playsound import playsound
import time


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


    def beam_search(self, input_seqs, beam_width=3):
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
        initial_beams = [BeamState(state=state, score=0.0, sequence=[input_seqs], logger=self.logger) for state in initial_states]

        beam_states = initial_beams

        # Perform beam search
        for _ in range(self.max_seq_length):
            new_beam_states = []
            for state in beam_states:
                if state.sequence[-1][-1] == self.tokenizer.end_token:
                    # If the hypothesis ends, add it to the final hypotheses
                    new_beam_states.append(state)
                else:
                    # Generate next token probabilities and states for all input sequences
                    decoder_input_token = tf.constant([[self.tokenizer.word_index[state.sequence[-1]]]] * len(input_seqs))
                    decoder_input_token = pad_sequences(decoder_input_token, maxlen=self.max_seq_length, padding='post', truncating='post')
                    
                    decoder_input = tf.constant(decoder_input_token, dtype=tf.float32)

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
        self.model_filename = "chatbot_model.keras"
        self.tokenizer_save_path = "chatBotTokenizer.pkl"
        self.tokenizer = None
        self.logger = self.setup_logger()  # Initialize your logger here
        self.embedding_dim = 100  # Define the embedding dimension here HAS TO BE SAME AS MAX_SEQ_LENGTH
        self.max_seq_length = 100  # Replace with your desired sequence length
        self.learning_rate = 0.001
        self.batch_size = 256
        self.epochs = 10
        self.vocabularyList = ['<PAD>', '<start>', '<end>', '<OOV>']
        self.max_vocab_size = len(self.vocabularyList)
        self.lstm_units = 2457
        self.perceivedMax = 10000
        self.speakerList = []
        self.encoder_model = None
        self.encoder_inputs = Input(shape=(self.max_seq_length,))
        self.decoder_inputs = Input(shape=(self.max_seq_length,))
        self.decoder_outputs = None
        self.decoder_model = None
        self. encoder_filename = "encoder.keras"
        self.decoder_filename = "decoder.keras"

        # Import Speakers
        with open('trained_speakers.txt', 'r') as file:
            self.speakerList = file.read().splitlines()

        if os.path.exists(self.tokenizer_save_path):
            with open(self.tokenizer_save_path, 'rb') as tokenizer_load_file:
                self.tokenizer = pickle.load(tokenizer_load_file)
                self.all_vocab_size = self.tokenizer.num_words
                self.logger.info("Tokenizer loaded successfully.")
                print("Number of words in loaded tokenizer:", len(self.tokenizer.word_index))
        else:
            self.logger.warning("Tokenizer not found, making now...  ")
            self.tokenizer = Tokenizer(num_words=None)  # Initialize the Tokenizer

            # Save '<OOV>', '<start>', and '<end>' to word index
            self.tokenizer.num_words = 0
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
        # Add '<start>' to beginning and '<end>'
        cleaned_text = f"<start> {cleaned_text} <end>"
        return cleaned_text.lower()


    def save_tokenizer(self, texts=None):
        if self.tokenizer:
            if texts:
                for token in texts:
                    if token not in self.tokenizer.word_index:
                        if token not in self.vocabularyList:
                            self.all_vocab_size += 1
                            self.tokenizer.num_words += 1
                            self.tokenizer.word_index[token] = self.tokenizer.num_words
                            print(f"Word: {token}\nIndex: {self.tokenizer.num_words}")
                            self.vocabularyList.append(token)

                self.tokenizer.fit_on_texts(texts)

            with open(self.tokenizer_save_path, 'wb') as tokenizer_save_file:
                pickle.dump(self.tokenizer, tokenizer_save_file)

            self.tokenizer.num_words = len(self.tokenizer.word_index)

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
        encoder_embedding = Embedding(input_dim=self.perceivedMax, output_dim=self.embedding_dim, input_length=max_seq_length)(self.encoder_inputs)
        encoder_lstm, state_h, state_c = LSTM(units=lstm_units, return_sequences=True, return_state=True, dropout=0.15, recurrent_dropout=0.1)(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_embedding = Embedding(input_dim=self.perceivedMax, output_dim=self.embedding_dim, input_length=max_seq_length)(self.decoder_inputs)
        decoder_lstm = LSTM(units=lstm_units, return_sequences=True, return_state=True, dropout=0.15, recurrent_dropout=0.1)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        decoder_dense = Dense(units=self.perceivedMax, activation='softmax')
        self.decoder_outputs = decoder_dense(decoder_outputs)

        # Create the model
        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

        # Inference
        self.encoder_model = Model(self.encoder_inputs, encoder_states)

        self.decoder_state_input_h = Input(shape=(lstm_units,))
        self.decoder_state_input_c = Input(shape=(lstm_units,))
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=self.decoder_states_inputs)
        self.decoder_states = [state_h, state_c]

        self.decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_model = Model([self.decoder_inputs] + self.decoder_states_inputs, [self.decoder_outputs] + self.decoder_states)

        # Compilation
        self.model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    def train_model(self, input_texts, target_texts, conversation_id, speaker):
        input_texts = [self.preprocess_text(phrase) for phrase in input_texts]
        target_texts = [self.preprocess_text(phrase) for phrase in target_texts]

        # Save the speakerList
        with open('trained_speakers.txt', 'a') as file:
            file.write(f'{speaker}\n')

        self.logger.info(f"Training Model... \nConversationID:  {conversation_id}")

        if self.corpus is None or self.tokenizer is None:
            raise ValueError("Corpus or tokenizer is not initialized.")

        # save_tokenizer and update num_words before training
        for lines in input_texts:
            self.save_tokenizer([word for word in lines.split(' ')])

        print(f"Num Words:  {self.tokenizer.num_words}")
        print(f"All Index:  {len(self.tokenizer.word_index)}")
        print(f"Length VocabList:  {len(self.vocabularyList)}")

        # Debug log line
        # self.logger.info(f"Word Counts:  {self.tokenizer.word_counts}")
        # self.logger.info(f"Tokenizer Config:  {self.tokenizer.get_config()}")

        # Preprocess the training data using the tokenizer
        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        # print(f"Input Seq:  {np.shape(input_sequences)}")
        padded_input_sequences = pad_sequences(input_sequences, maxlen=self.max_seq_length, padding='post')
        # print(f"Padded Input Seq:  \n{padded_input_sequences}")
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)
        # print(f"Target Seq:  {np.shape(target_sequences)}")
        padded_target_sequences = pad_sequences(target_sequences, maxlen=self.max_seq_length, padding='post')
        # print(f"Padded Target Seq:  \n{padded_target_sequences}")

        # Split this speaker's data into training and test sets
        train_input, test_input, train_target, test_target = train_test_split(padded_input_sequences, padded_target_sequences, test_size=0.2, random_state=86)

        # Print Model Summary
        # print(self.model.summary())

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

        playsound("AlienNotification.mp3")
        time.sleep(10)


    def save_model(self):
        self.logger.info("Saving Model...")
        if self.model:
            self.model.save(self.model_filename)
            self.encoder_model.save(self.encoder_filename)
            self.decoder_model.save("decoder.keras")
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

        elif not os.path.exists(model_filename):
            self.logger.warning("No saved model found... Making now...  ")
            # Build the model (if not already built)
            if self.model is None:
                self.build_model()


    def generate_response_with_beam_search(self, user_input, beam_width=3, batch_size=None):
        # Preprocess user input
        user_input = self.preprocess_text([user_input]).split(" ")
        user_input_seq = self.tokenizer.texts_to_sequences([user_input])
        user_input_seq = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post')

        self.tokenizer.start_token = self.vocabularyList[0]
        self.tokenizer.end_token = self.vocabularyList[1]

        reverse_target_char_index = dict(map(reversed, self.tokenizer.word_index.items()))

        beamHelper = BeamSearchHelper(self.model, self.tokenizer, self.max_seq_length)

        # Perform beam search
        response_sequences = beamHelper.beam_search(user_input_seq, beam_width=beam_width)

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

    def generate_response(self, user_input):
        start_token_id = 1
        end_token_id = 2

        user_input = self.preprocess_text(user_input)

        # Texts to seq for evaluation
        user_input_seq = self.tokenizer.texts_to_sequences([user_input])[0]

        user_input_seq = np.array([user_input_seq])
        
        padded_input_sequences = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post', truncating='post')
        print(padded_input_sequences)

        # Encode the input sequence using the chatbot model (encoder and decoder together)
        chatbot_output, _ = self.model.predict(padded_input_sequences)

        # Initialize the decoder input with a start token
        decoder_input = np.array([[start_token_id]])

        # List to store the generated output tokens
        output_tokens = []

        # Perform decoding step by step
        for _ in range(max_sequence_length):  # Adjust max_sequence_length as needed
            # Use the chatbot model to predict the next token
            chatbot_output, _ = self.model.predict([decoder_input, padded_input_sequences])

            # Extract the decoder output (assuming it's the first part of the chatbot output)
            decoder_output = chatbot_output[:, :max_sequence_length, :]

            # Print the structure of the decoder_output
            print("Decoder Output Shape:", decoder_output.shape)

            # Get the index of the predicted token
            predicted_token_index = np.argmax(decoder_output)

            # Append the predicted token to the output
            output_tokens.append(predicted_token_index)

            # Set the current predicted token as the input for the next decoding step
            decoder_input = np.array([[predicted_token_index]])

            # Check if the end token is predicted
            if predicted_token_index == end_token_id:
                break

        # Convert the output tokens to words using your tokenizer or index-to-word mapping
        predicted_output_words = [self.tokenizer.index_word.get(token, '<OOV>') for token in output_tokens]
        response_string = " ".join(response_texts)

        return response_string