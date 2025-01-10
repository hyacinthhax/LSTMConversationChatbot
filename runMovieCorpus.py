import os
import tensorflow
from chatbotTrainer import processed_dialogs
from playsound3 import playsound
from chatbotTrainer import ChatbotTrainer
import time
import random
import pdb
import convokit


# Larger list should be last so the fraction is converted to percent(bad_count/total_count)
def runningPercent(list1, list2):
    if list1 > 0 and list2 > 0:
        x = list1 / list2
        percentage = x * 100
        percentage = round(percentage, 2)

        return percentage

    elif list1 == 0:
        percentage = 0.0
        return percentage

def run(chatbot_trainer, user_choice, topConvo=0, top_num=0):
    # topConvo is a larger buffer for models that may take longer to learn. top_num is the default
    # All input/target lists are for scripts if ran for context
    counter = 0
    bad_count = 0
    all_input_texts = []
    all_target_texts = []
    speakerList = []
    speakerListData = None
    troubleListData = None
    troubleList = []
    allTogether = []
    choices_yes = ["yes", "ya", "yeah", "yessir", "yesir", "y", "ye"]
    exit_commands = ["exit", "quit", "stop", "x", ""]

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

        speakerList = sorted(speakerList)
        return speakerList

    def resetTroubled():
        os.remove('troubled_speakers.txt')
        with open('troubled_speakers.txt', 'w') as f:
            f.write("")

    # We Reset the file after trouble list reset(Trouble List should be empty before and after this step)
    resetTroubled()

    # We clean up the trained
    speakerList = cleanupTrained(speakerListData)

    def resetTogether(speakerList, troubleListData):
        for speakers in speakerList:
            if speakers not in allTogether:
                allTogether.append(str(speakers))
        for speakers in troubleListData:
            if speakers not in allTogether:
                allTogether.append(str(speakers))
        allTogetherSorted = sorted(allTogether)

        return allTogetherSorted

    all_listed = []
    # Debug Lines
    # pdb.set_trace()
    # print(list(speakerList))

    for x in range(len(processed_dialogs.keys())):
        topConvo += 1
        counter += 1
        randomconvo = random.randint(1, len(processed_dialogs.keys()))
        speaker = str(randomconvo)
        dialog_pairs = processed_dialogs[speaker]
        if speaker not in speakerList:
            conversation_id = int(speaker)
            if conversation_id > top_num:
                top_num = conversation_id
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
                chatbot_trainer.train_model(speaker_input_texts, speaker_target_texts, str(conversation_id), speaker)
                playsound("AlienNotification.mp3")
                if speaker not in speakerList and speaker not in runningTrouble:
                    speakerList.append(speaker)

                    with open("trained_speakers.txt", 'a') as f:
                        f.write(f"{speaker}\n")

                    continue

                elif runningTrouble.count(speaker) >= chatbot_trainer.early_patience:
                    # We update troubleList here on going for each speaker not saved to speakerList
                    if speaker not in troubleList:
                        bad_count += 1
                        troubleList.append(speaker)

                    with open("troubled_speakers.txt", 'a') as f:
                        f.write(f"{speaker}\n")

                allTogether = resetTogether(speakerList, troubleList)
                percent_running = runningPercent(bad_count, topConvo)
                if percent_running is None:
                    percent_running = 0.0
                chatbot_trainer.logger.info(f"Running Percentage Failure: {percent_running}%")

                # We check for speaker vs top num achieved successfully in speakerList 
                if counter < topConvo:
                    print(f"Conversations Completed Total:  {counter}\n Don't quit you haven't surpassed {speaker}/{top_num}\n Trouble List will be reset if this is an automated run...")
                elif counter >= topConvo:
                    print(f"Now is the time to quit if need be...  ")
                    playsound("AlienNotification.mp3")

                if percent_running is not None:
                    if percent_running > 85.0:
                        flip_token = random.randint(0, 1)
                        if flip_token == 1:
                            print("Restarting to Tackle Trouble List...  ")
                            resetTroubled()
                            return run(chatbot_trainer, user_choice, topConvo, top_num)
                        elif flip_token == 0:
                            continue

                input("\nEnter to Continue...  ")

            elif user_choice.lower() in exit_commands:
                quit()

            elif user_choice.lower() not in choices_yes and user_choice.lower() not in exit_commands:
                chatbot_trainer.train_model(speaker_input_texts, speaker_target_texts, str(conversation_id), speaker)
                if speaker not in speakerList and speaker not in runningTrouble:
                    speakerList.append(speaker)

                    with open("trained_speakers.txt", 'a') as f:
                        f.write(f"{speaker}\n")

                    continue

                elif runningTrouble.count(speaker) >= chatbot_trainer.early_patience:
                    # We update troubleList here on going for each speaker not saved to speakerList
                    if speaker not in troubleList:
                        bad_count += 1
                        troubleList.append(speaker)

                    with open("troubled_speakers.txt", 'a') as f:
                        f.write(f"{speaker}\n")

                # Find Top Convo
                allTogether = resetTogether(speakerList, troubleList)
                percent_running = runningPercent(bad_count, topConvo)
                if percent_running is None:
                    percent_running = 0.0
                chatbot_trainer.logger.info(f"Running Percentage Failure: {percent_running}%")

                # We check for speaker vs top num achieved successfully in speakerList
                if counter < topConvo:
                    print(
                        f"Conversations Completed Total:  {counter}\n Don't quit you haven't surpassed {counter}/{top_num}\n Trouble List will be reset if this is an automated run...")
                elif counter >= topConvo:
                    print(f"Now is the time to quit if need be...  ")
                    time_sleep = 10
                    for x in range(time_sleep):
                        time.sleep(1)
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print(f"Next convo in:{time_sleep-x}")

                if percent_running is not None:
                    if percent_running > 85.0:
                        flip_token = random.randint(0, 1)
                        if flip_token == 1:
                            print("Restarting to Tackle Trouble List...  ")
                            resetTroubled()
                            return run(chatbot_trainer, user_choice, topConvo, top_num)
                        elif flip_token == 0:
                            continue

        else:
            print(f"{speaker} Skipped for being on List.")
            continue


def cleanupTroubled():
    tempBin = []
    with open('troubled_speakers.txt', 'r') as fr:
        data = fr.readlines()
        for lines in data:
            if lines not in tempBin:
                tempBin.append(str(lines).strip('\n'))

    tempBin = sorted(tempBin)
    with open('troubled_speakers.txt', 'w') as fw:
        fw.write("")
        for troubled in tempBin:
            fw.write(f"{troubled}\n")


def main():
    print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    chatbot_trainer = ChatbotTrainer()

    # Initialize the corpus (Needed for convo-kit to initialize)
    corpus_path = 'D:\\movie-corpus'
    chatbot_trainer.load_corpus(corpus_path)

    try:
        cleanupTroubled()
        user_choice = input(f"Run Supervised?({chatbot_trainer.model_filename})\n>")
        run(chatbot_trainer, user_choice)

    except Exception as e:
        chatbot_trainer.logger.warning(e)

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()