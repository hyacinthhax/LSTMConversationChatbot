import os
import convokit
import re

class DialogProcessor:
    def __init__(self):
        self.corpus = None

    def load_corpus(self, corpus_path):
        self.corpus = convokit.Corpus(filename=corpus_path)

    @staticmethod
    def preprocess_text(text):
        cleaned_text = text.replace('"', '')
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text.lower()

    def process_dialogs(self):
        if self.corpus is None:
            raise ValueError("Corpus is not loaded.")

        dialogue_data = {}  # Dictionary format: {conversation_id: [(user_input, chatbot_response), ...]}

        for conversation_id in self.corpus.get_conversation_ids():
            print(conversation_id)
            conversation = self.corpus.get_conversation(conversation_id)
            dialog_pairs = []

            for i in range(len(conversation.utterances) - 1):
                user_input = self.preprocess_text(conversation.utterances[i].text)
                chatbot_response = self.preprocess_text(conversation.utterances[i + 1].text)
                dialog_pairs.append((user_input, chatbot_response))

            dialogue_data[conversation_id] = dialog_pairs

        return dialogue_data

if __name__ == "__main__":
    dialog_processor = DialogProcessor()

    corpus_path = "C:\\Users\\admin\\Desktop\\movie-corpus"
    dialog_processor.load_corpus(corpus_path)

    processed_dialogs = dialog_processor.process_dialogs()

    # Save processed dialogs as a Python module
    save_path = "processed_dialogs.py"
    with open(save_path, "w") as f:
        f.write("processed_dialogs = {\n")
        for conversation_id, dialog_pairs in processed_dialogs.items():
            f.write(f"    '{conversation_id}': [\n")
            for user_input, chatbot_response in dialog_pairs:
                f.write(f"('{user_input}', '{chatbot_response}'),\n")
            f.write("],\n")
        f.write("}\n")
