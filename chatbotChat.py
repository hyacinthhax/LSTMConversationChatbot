import re
from chatbotTrainer import ChatbotTrainer  # Import the ChatbotTrainer class


def main():
    # Initialize the chatbot
    chatbot_trainer = ChatbotTrainer()

    # Ensure the model and tokenizer are loaded
    if chatbot_trainer.model is None:
        chatbot_trainer.load_model_file()

    print("Chatbot is ready. Type 'exit' to end the conversation.")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                print("Chatbot: Please say something, I'm here to help!")
                continue

            if user_input.lower() == "exit":
                print("Chatbot: Goodbye! Have a great day!")
                break

            # Generate a response
            response = chatbot_trainer.generate_response(user_input)

            # Handle empty or invalid responses
            if not response or response.strip() == "":
                response = "I'm sorry, I don't have a response for that."

            print(f"Alan: {response}")
        except Exception as e:
            print(f"Chatbot: An error occurred while generating a response. ({str(e)})")


# Run the chatbot if the script is executed directly
if __name__ == "__main__":
    main()
