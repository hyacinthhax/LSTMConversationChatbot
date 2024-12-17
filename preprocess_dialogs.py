import os
import convokit
import re
import json


class DialogProcessor:
    def __init__(self):
        self.corpus = None

    def load_corpus(self, corpus_path):
        # Load the corpus from a local file or download it
        self.corpus = convokit.Corpus(filename=convokit.download('movie-corpus'))

    @staticmethod
    def preprocess_text(text):
        # Clean text by removing tags, quotes, and extra spaces
        cleaned_text = re.sub(r'<u>.*?</u>', '', text)
        cleaned_text = cleaned_text.replace('"', '')
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text.lower()

    def group_conversations(self):
        if self.corpus is None:
            raise ValueError("Corpus is not loaded.")

        grouped_dialogues = {}  # Dictionary format: {conversation_id: [{"person1": "text", "person2": "response"}, ...]}
        misc_dialogues = []
        current_dialog_id = 1  # Start numbering from 1

        for conversation_id in self.corpus.get_conversation_ids():
            conversation = self.corpus.get_conversation(conversation_id)
            utterances = conversation.get_utterance_ids()
            current_dialog = []

            for i in range(len(utterances) - 1):
                user_utterance = self.corpus.get_utterance(utterances[i])
                response_utterance = self.corpus.get_utterance(utterances[i + 1])

                user_input = self.preprocess_text(user_utterance.text)
                chatbot_response = self.preprocess_text(response_utterance.text)

                # Append person1 and person2 conversation format
                current_dialog.append({"person1": user_input, "person2": chatbot_response})

                # Check if the next response is from the same speaker or the last in the conversation
                if user_utterance.speaker.id == response_utterance.speaker.id or i == len(utterances) - 2:
                    # Save dialog only if it has at least 4 exchanges
                    if len(current_dialog) >= 4:
                        grouped_dialogues[current_dialog_id] = current_dialog
                        current_dialog_id += 1
                    else:
                        misc_dialogues.append(current_dialog)
                    current_dialog = []

        # Add misc dialogues to list 0
        grouped_dialogues[0] = misc_dialogues

        return grouped_dialogues

    def save_grouped_conversations(self, grouped_dialogues, save_path="processed_dialogs.json"):
        # Save the grouped dialogues to a JSON file for better compatibility
        with open(save_path, "w") as f:
            json.dump(grouped_dialogues, f, indent=4, ensure_ascii=False)
        print(f"Grouped conversations saved to {save_path}")


if __name__ == "__main__":
    dialog_processor = DialogProcessor()

    # Specify the corpus path or use 'movie-corpus' to download
    corpus_path = "D:\\movie-corpus"  # Replace with your actual path if local
    dialog_processor.load_corpus(corpus_path)

    grouped_dialogues = dialog_processor.group_conversations()

    # Save the processed dialogues as JSON
    save_path = "processed_dialogs.json"
    dialog_processor.save_grouped_conversations(grouped_dialogues, save_path)
