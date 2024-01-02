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
import pdb


def run(chatbot_trainer, speakerList, all_input_texts=[], all_target_texts=[]):
    for speaker, dialog_pairs in dialog_data.items():
        if speaker not in speakerList:
            conversation_id = f"'{speaker}'"
            print(f"Speaker: {conversation_id}")

            # Initialize lists for this speaker's data
            speaker_input_texts = []
            speaker_target_texts = []

            for input_text, target_text in dialog_pairs:
                if input_text != "":

                    # pdb.set_trace()
                    # Tokenize the input and target text into words
                    input_words = chatbot_trainer.preprocess_text(input_text).split()
                    target_words = chatbot_trainer.preprocess_text(target_text).split()

                    # Add unique words from input text to the vocabulary
                    input_list = []
                    for word in input_words:
                        input_list.append(word)

                    input_words = ' '.join(input_list)

                    # Add unique words from target text to the vocabulary
                    target_list = []
                    for word in target_words:
                        target_list.append(word)

                    target_words = ' '.join(target_list)

                    speaker_input_texts.append(input_words)
                    all_input_texts.append(input_words)
                    speaker_target_texts.append(target_words)
                    all_target_texts.append(target_words)

            # Train the model using the preprocessed training data for this speaker
            chatbot_trainer.train_model(speaker_input_texts, speaker_target_texts, conversation_id, speaker)

        else:
            print(f"{speaker} Skipped for being on List.")
            continue


def main():
    speakerList = []
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    chatbot_trainer = ChatbotTrainer()

    # Initialize the corpus
    corpus_path = "C:\\Users\\admin\\Desktop\\movie-corpus"
    chatbot_trainer.load_corpus(corpus_path)

    try:
        run(chatbot_trainer, speakerList)

    except Exception as e:
        print(e)
        chatbot_trainer.max_vocab_size += 10000
        run(chatbot_trainer, speakerList)


    


if __name__ == "__main__":
    main()
