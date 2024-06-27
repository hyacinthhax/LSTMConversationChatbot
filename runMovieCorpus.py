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


def runningPercent(list1, list2):
    if len(list1) > 0 and len(list2) > 0:
        x = len(list1) / len(list2)
        percentage = x * 100
        percentage = round(percentage, 2)
        return percentage

    elif len(list1) == 0:
        percentage = 0.0
        return percentage

def run(chatbot_trainer, user_choice):
    # All input/target lists are for scripts if ran for context
    all_input_texts = []
    all_target_texts = []
    speakerList = []
    speakerListData = None
    troubleListData = None
    troubleList = []
    allTogether = []
    choices_yes = ["yes", "ya", "yeah", "yessir", "yesir", "y", "ye"]

    # Import Speakers
    with open('trained_speakers.txt', 'r') as file:
        speakerListData = file.read().splitlines()

    with open('troubled_speakers.txt', 'r') as file:
        troubleListData = file.read().splitlines()


    def cleanupTrained(speakerList):
        for data in speakerList:
            data = data.strip('\n')
            if data not in speakerList and data not in troubleListData:
                speakerList.append(data)
                with open('trained_speakers.txt', 'w') as f:
                    for speakers in speakerList:
                        f.write(f"{speakers}\n")

        speakerList = sorted(speakerList, key=int)
        return speakerList

    def resetTroubled():
        os.remove('troubled_speakers.txt')
        with open('troubled_speakers.txt', 'w') as f:
            f.write("")

    # We Reset the file after trouble list reset(Trouble List should be empty before and after this step)
    resetTroubled()

    # We clean up the trained
    speakerList = cleanupTrained(speakerListData)

    def resetTogether(speakerList, troubleListData, topConvo):
        topConvo = 0
        for speakers in speakerList:
            if speakers not in allTogether:
                topConvo += 1
                allTogether.append(speakers.strip('\n'))
        for speakers in troubleListData:
            if speakers not in allTogether:
                topConvo += 1
                allTogether.append(speakers.strip('\n'))
        allTogetherSorted = sorted(allTogether, key=int)

        return allTogetherSorted

    # Debug Line
    # print(list(speakerList))

    for speaker, dialog_pairs in processed_dialogs.items():
        if speaker not in speakerList:
            conversation_id = f"'{speaker}'"
            print(f"Speaker: {speaker}")
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

            runningTrouble = chatbot_trainer.troubleList
            # Train the model using the preprocessed training data for this speaker
            if user_choice.lower() in choices_yes:
                chatbot_trainer.train_model(speaker_input_texts, speaker_target_texts, conversation_id, speaker)
                # playsound("AlienNotification.mp3")    # Not Working due to error in playsound(Works once, then fails next, might need to stop sound)
                if speaker not in speakerList and speaker not in runningTrouble:
                    speakerList.append(speaker)

                    with open("trained_speakers.txt", 'a') as f:
                        f.write(f"{speaker}\n")

                # We update troubleList here on going for each speaker not saved to speakerList
                if len(runningTrouble) != 0:
                    if speakers not in troubleList:
                        troubleList.append(speakers)

                if speaker in troubleList:
                    with open("troubled_speakers.txt", 'a') as f:
                        f.write(f"{speaker}\n")

                # Find Top Convo
                topConvo = int(cleanupTrained(speakerList)[-1])
                allTogether = resetTogether(speakerList, troubleList, topConvo)
                topConvo = len(allTogether)
                percent_running = runningPercent(troubleList, allTogether)
                chatbot_trainer.logger.info(f"Running Percentage Failure: {percent_running}%")

                # We check for speaker vs top num achieved successfully in speakerList 
                if int(speaker.strip('\n')) < topConvo:
                    print(f"Conversations Completed Total:  {topConvo}\n Don't quit you haven't surpassed {speaker}/{allTogether[-1]}\n Trouble List will be reset if this is an automated run...")
                elif int(speaker.strip('\n')) > topConvo:
                    print(f"Now is the time to quit if need be...  ")                

                if percent_running != None:
                    if percent_running > 50.0:
                        troubleList = []
                        print("Restarting to Tackle Trouble List...  ")
                        resetTroubled()
                        return run(chatbot_trainer, user_choice)

                input("\nEnter to Continue...  ")

            elif user_choice.lower() not in choices_yes:
                chatbot_trainer.train_model(speaker_input_texts, speaker_target_texts, conversation_id, speaker)
                if speaker not in speakerList and speaker not in runningTrouble:
                    speakerList.append(speaker)

                    with open("trained_speakers.txt", 'a') as f:
                        f.write(f"{speaker}\n")

                for speakers in runningTrouble:
                    if speakers not in troubleList:
                        troubleList.append(speakers)

                if len(runningTrouble) != 0:
                    with open("troubled_speakers.txt", 'a') as f:
                        f.write(f"{speaker}\n")

                # Find Top Convo
                topConvo = int(cleanupTrained(speakerList)[-1])
                allTogether = resetTogether(speakerList, troubleList, topConvo)
                topConvo = len(allTogether)
                percent_running = runningPercent(troubleList, allTogether)
                chatbot_trainer.logger.info(f"Running Percentage Failure: {percent_running}%")
                if int(speaker.strip('\n')) < topConvo:
                    print(f"Conversations Completed Total:  {topConvo}\n Don't quit you haven't surpassed {speaker}/{allTogether[-1]}\n Trouble List will be reset if this is an automated run...")
                elif int(speaker.strip('\n')) >= topConvo:
                    print(f"Now is the time to quit if need be...  ")

                if percent_running != None:
                    if percent_running > 50.0:
                        troubleList = []
                        print("Restarting to Tackle Trouble List...  ")
                        resetTroubled()
                        return run(chatbot_trainer, user_choice)

                time.sleep(10)

            else:
                print(f"\nSkipped {speaker} for not providing enough data...  \n")

        else:
            cleanupTroubled()
            print(f"{speaker} Skipped for being on List.")
            continue


def cleanupTroubled():
    tempBin = []
    with open('troubled_speakers.txt', 'r') as fr:
        data = fr.readlines()
        for lines in data:
            if lines not in tempBin:
                tempBin.append(lines.strip('\n'))

    tempBin = sorted(tempBin, key=int)
    with open('troubled_speakers.txt', 'w') as fw:
        fw.write("")
        for troubled in tempBin:
            fw.write(f"{troubled}\n")


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    chatbot_trainer = ChatbotTrainer()

    # Initialize the corpus (Needed for convo-kit to initialize)
    corpus_path = "E:\\movie-corpus"
    chatbot_trainer.load_corpus(corpus_path)

    try:
        cleanupTroubled()
        user_choice = input(f"Run Supervised?({chatbot_trainer.model_filename})\n>")
        run(chatbot_trainer, user_choice)

    except Exception as e:
        chatbot_trainer.logger.warning(e)

    


if __name__ == "__main__":
    main()