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
from processed_dialogs import processed_dialogs
from playsound import playsound
from chatbotTrainer import ChatbotTrainer
import time
import pdb


def run(chatbot_trainer):
    speakerNumber = 0
    all_input_texts = []
    all_target_texts = []
    speakerList = []
    speakerListData = None
    troubleListData = None
    # Import Speakers
    with open('trained_speakers.txt', 'r') as file:
        speakerListData = file.read().splitlines()

    with open('troubled_speakers.txt', 'r') as file:
        troubleListData = file.read().splitlines()

    for data in speakerListData:
        if data not in troubleListData:
            speakerList.append(data)

    # Debug Line
    print(list(speakerList))

    choices_yes = ["yes", "ya", "yeah", "yessir", "yesir", "y", "ye"]
    user_choice = input(f"Run Supervised?({chatbot_trainer.model_filename})\n>")
    for speaker, dialog_pairs in processed_dialogs.items():
        if speaker not in speakerList:
            conversation_id = f"'{speaker}'"
            print(f"Speaker: {conversation_id}")
            # Initialize lists for this speaker's data
            speaker_input_texts = []
            speaker_target_texts = []

            # Input conversation data into input and target data from dialog pairs
            for input_text, target_text in dialog_pairs:
                if input_text != "" and target_text != "":
                    speaker_input_texts.append(input_text)
                    all_input_texts.append(input_text)
                    speaker_target_texts.append(target_text)
                    all_target_texts.append(target_text)

            # Only train if conversation has more than 3 input/target texts. (Either or)
            if len(speaker_input_texts) > 3:
                # Train the model using the preprocessed training data for this speaker
                if user_choice.lower() in choices_yes:
                    chatbot_trainer.train_model(speaker_input_texts, speaker_target_texts, speakerNumber, speaker)
                    speakerNumber +=1
                    print(f"Conversations Completed Total:  {speakerNumber}")
                    # playsound("AlienNotification.mp3")    # Not Working due to error in playsound(Works once, then fails next, might need to stop sound)
                    if speaker not in speakerList:
                        speakerList.append(speaker)
                        with open("trained_speakers.txt", 'a') as f:
                            f.write(f"{speaker}\n")
                    input("\nEnter to Continue...  ")
                else:
                    chatbot_trainer.train_model(speaker_input_texts, speaker_target_texts, speakerNumber, speaker)
                    speakerNumber +=1
                    if speaker not in speakerList:
                        speakerList.append(speaker)
                        with open("trained_speakers.txt", 'a') as f:
                            f.write(f"{speaker}\n")
                    print(f"Conversations Completed Total:  {speakerNumber}")

            else:
                print(f"\nSkipped {speaker} for not providing enough data...  \n")

        else:
            print(f"{speaker} Skipped for being on List.")
            continue


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    chatbot_trainer = ChatbotTrainer()

    # Initialize the corpus (Needed for convo-kit to initialize)
    corpus_path = "E:\\movie-corpus"
    chatbot_trainer.load_corpus(corpus_path)

    try:
        run(chatbot_trainer)

    except Exception as e:
        print(e)

    


if __name__ == "__main__":
    main()