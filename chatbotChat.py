import re
import numpy as np
import pickle
from chatbotTrainer import ChatbotTrainer  # Import the ChatbotTrainer class


chatbot_trainer = ChatbotTrainer()
print("Chatbot is ready. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    if not user_input or user_input == "":
        print("Chatbot: I'm sorry, I don't understand your input.")
        continue
    
    response = chatbot_trainer.generate_response(user_input)

    print(f"Alan: {response}")
