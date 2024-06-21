import os
import convokit
import re

class DialogProcessor:
    def __init__(self):
        self.corpus = None

    def load_corpus(self, corpus_path):
        self.corpus = convokit.Corpus(filename=convokit.download('movie-corpus') if corpus_path == 'movie-corpus' else corpus_path)

    @staticmethod
    def preprocess_text(text):
        # Remove <u> and </u> tags and their contents
        cleaned_text = re.sub(r'<u>.*?</u>', '', text)

        # Remove double quotes from the text
        cleaned_text = cleaned_text.replace('"', '')

        # Remove multiple spaces
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text.lower()

    def group_conversations(self):
        if self.corpus is None:
            raise ValueError("Corpus is not loaded.")

        grouped_dialogues = {}  # Dictionary format: {conversation_id: [(user_input, chatbot_response), ...]}

        for conversation_id in self.corpus.get_conversation_ids():
            conversation = self.corpus.get_conversation(conversation_id)
            utterances = conversation.get_utterance_ids()
            dialog_pairs = []
            current_dialog = []

            for i in range(len(utterances) - 1):
                user_utterance = self.corpus.get_utterance(utterances[i])
                response_utterance = self.corpus.get_utterance(utterances[i + 1])
                user_input = self.preprocess_text(user_utterance.text)
                chatbot_response = self.preprocess_text(response_utterance.text)

                if user_utterance.speaker.id != response_utterance.speaker.id:
                    current_dialog.append((user_input, chatbot_response))
                else:
                    if current_dialog:
                        if conversation_id not in grouped_dialogues:
                            grouped_dialogues[conversation_id] = []
                        grouped_dialogues[conversation_id].append(current_dialog)
                    current_dialog = [(user_input, chatbot_response)]

            if current_dialog:  # Add the last dialog in the sequence
                if conversation_id not in grouped_dialogues:
                    grouped_dialogues[conversation_id] = []
                grouped_dialogues[conversation_id].append(current_dialog)

        return grouped_dialogues

if __name__ == "__main__":
    dialog_processor = DialogProcessor()

    corpus_path = "movie-corpus"  # Use 'movie-corpus' to download and use the corpus
    dialog_processor.load_corpus(corpus_path)

    grouped_dialogues = dialog_processor.group_conversations()

    # Save processed dialogs as a Python module
    save_path = "processed_dialogs.py"
    with open(save_path, "w") as f:
        f.write("processed_dialogs = {\n")
        for conversation_id, dialog_groups in grouped_dialogues.items():
            f.write(f"    '{conversation_id}': [\n")
            for dialog in dialog_groups:
                for user_input, chatbot_response in dialog:
                    f.write(f"        ('{user_input}', '{chatbot_response}'),\n")
            f.write("    ],\n")
        f.write("}\n")
