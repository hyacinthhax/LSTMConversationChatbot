import pickle
from chatbotTrainer import ChatbotTrainer

# Assuming max_seq_length is an attribute in your ChatbotTrainer class
chatbot_model = ChatbotTrainer()
max_seq_length = chatbot_model.max_seq_length

print("Chatbot is ready. Type 'exit' to end the conversation.")
chatbot_model.load_model_file()

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    
    input_text = chatbot_model.preprocess_text(user_input)

    response_text = " ".join(chatbot_model.generate_response(input_text))
    response_text = response_text.replace("<start>", "").replace("<end>", "")

    print(f"Chatbot: {response_text}")
