import os

def setup_chatbot():
    # Install dependencies using pip
    os.system('pip install -r requirements.txt')

    print("Setup completed. You can now run the chatbot with 'python runMovieCorpus.py'.")

if __name__ == "__main__":
    setup_chatbot()