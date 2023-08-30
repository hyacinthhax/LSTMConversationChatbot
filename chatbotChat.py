import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class Chatbot:
    def __init__(self, model_path, tokenizer):
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = tokenizer

    def preprocess_text(self, text):
        cleaned_text = text.replace('"', '')
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text.lower()

    def generate_response(self, input_text, max_response_length=100):
        input_text = self.preprocess_text(input_text)
        input_seq = self.tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')
        response_seq = self.model.predict(input_seq)
        response_seq = np.argmax(response_seq, axis=-1)
        response_text = self.tokenizer.sequences_to_texts(response_seq)[0]
        return response_text

if __name__ == "__main__":
    model_path = "chatbot_model.h5"  # Path to your trained model
    max_seq_length = 100  # Maximum sequence length

    tokenizer = Tokenizer(oov_token="<OOV>")  # Use the same tokenizer parameters as during training
    chatbot = Chatbot(model_path, tokenizer)

    print("Chatbot is ready. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        response = chatbot.generate_response(user_input, max_response_length=max_seq_length)
        print(f"Chatbot: {response}")