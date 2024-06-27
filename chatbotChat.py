import os
import re
import numpy as np
import tensorflow
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
