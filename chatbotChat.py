import re
import numpy as np
import pickle
from chatbotTrainer import ChatbotTrainer  # Import the ChatbotTrainer class




model_path = "chatbot_model.h5"
max_seq_length = 100
chatbot_trainer = ChatbotTrainer()

# Load the saved tokenizer
tokenizer_load_path = "chatBotTokenizer.pkl"  # Update with the actual path to your saved tokenizer
with open(tokenizer_load_path, 'rb') as tokenizer_load_file:
    loaded_tokenizer = pickle.load(tokenizer_load_file)
    chatbot_trainer.tokenizer = loaded_tokenizer
    print("Number of words in loaded tokenizer:", len(chatbot_trainer.tokenizer.word_index))


chatbot_trainer.load_model()
print("Chatbot is ready. Type 'exit' to end the conversation.")


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    
    input_text = chatbot_trainer.preprocess_text(user_input)
    if not input_text:
        print("Chatbot: I'm sorry, I don't understand your input.")
        continue
    
    response = chatbot_trainer.generate_response(input_text)
    print(f"Chatbot: {response}")
