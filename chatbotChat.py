import re
import numpy as np
import pickle
from chatbot_trainer import ChatbotTrainer  # Import the ChatbotTrainer class

def preprocess_text(text):
    cleaned_text = text.replace('"', '')
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text.lower()

if __name__ == "__main__":
    model_path = "chatbot_model.h5"
    max_seq_length = 100
    
    # Load the saved tokenizer
    tokenizer_load_path = "chatBotTokenizer.pkl"  # Update with the actual path to your saved tokenizer
    with open(tokenizer_load_path, 'rb') as tokenizer_load_file:
        loaded_tokenizer = pickle.load(tokenizer_load_file)
        print("Number of words in loaded tokenizer:", len(loaded_tokenizer.word_index))
    
    chatbot_trainer = ChatbotTrainer()
    chatbot_trainer.load_model(model_path, loaded_tokenizer, max_seq_length)
    
    print("Chatbot is ready. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        input_text = preprocess_text(user_input)
        if not input_text:
            print("Chatbot: I'm sorry, I don't understand your input.")
            continue
        
        response = chatbot_trainer.generate_response(input_text)
        print(f"Chatbot: {response}")