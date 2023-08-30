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

        dialogue_data = {}  # Dictionary format: {speaker: [(input, target), ...]}

        for speaker in self.corpus.get_speaker_ids():
            print(speaker)
            speaker_utterance_ids = self.corpus.get_utterance_ids(
                selector=lambda utt: utt.speaker.id == speaker
            )
            if len(speaker_utterance_ids) < 2:
                continue

            speaker = str(speaker)

            dialogue_pairs = []
            for i in range(len(speaker_utterance_ids) - 1):
                input_text = self.preprocess_text(
                    self.corpus.get_utterance(speaker_utterance_ids[i]).text
                )
                target_text = self.preprocess_text(
                    self.corpus.get_utterance(speaker_utterance_ids[i + 1]).text
                )
                dialogue_pairs.append((input_text, target_text))

            dialogue_data[speaker] = dialogue_pairs

        return dialogue_data

if __name__ == "__main__":
    dialog_processor = DialogProcessor()

    corpus_path = "C:\\Users\\admin\\Desktop\\movie-corpus"
    dialog_processor.load_corpus(corpus_path)

    processed_dialogs = dialog_processor.process_dialogs()

    # Save processed dialogs as a Python module
    save_path = "processed_dialogs.py"
    with open(save_path, "w") as f:
        f.write("dialog_data = {")
        for key, data in processed_dialogs.items():
            f.write(f"'{key}'" + ":" + f"{data}" + ",\n")
        f.write("}")
