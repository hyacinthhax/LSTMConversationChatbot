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


def run(chatbot_trainer):
    all_input_texts = []
    all_target_texts = []
    for speaker, dialog_pairs in dialog_data.items():
        if speaker not in chatbot_trainer.speakerList:
            conversation_id = f"'{speaker}'"
            print(f"Speaker: {conversation_id}")

            # Initialize lists for this speaker's data
            speaker_input_texts = []
            speaker_target_texts = []

            for input_text, target_text in dialog_pairs:
                if input_text != "" and target_text != "":
                    # pdb.set_trace()

                    speaker_input_texts.append(input_text)
                    all_input_texts.append(input_text)
                    speaker_target_texts.append(target_text)
                    all_target_texts.append(target_text)

            if len(speaker_input_texts) > 3:
                conversation = []
                for i, input_text in enumerate(speaker_input_texts):
                    conversation.append(input_text)
                    target_text = speaker_target_texts[i]  # Get the corresponding target text
                    if len(conversation) > 3:
                        chatbot_trainer.train_model(conversation, [target_text], conversation_id, speaker)
                # input("Enter to continue...  ")   # This is for supervised training and testing

            else:
                print(f"\nSkipped {speaker} for not providing enough data...  \n")

        else:
            print(f"{speaker} Skipped for being on List.")
            continue


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    chatbot_trainer = ChatbotTrainer()

    # Initialize the corpus
    corpus_path = "C:\\Users\\admin\\Desktop\\movie-corpus"
    chatbot_trainer.load_corpus(corpus_path)

    try:
        run(chatbot_trainer)

    except Exception as e:
        print(e)

    


if __name__ == "__main__":
    main()