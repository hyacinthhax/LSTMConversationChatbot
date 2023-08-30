import unittest
from unittest.mock import patch, MagicMock
from chatbot_trainer import ChatbotTrainer

class TestChatbotTrainer(unittest.TestCase):

    def setUp(self):
        self.chatbot_trainer = ChatbotTrainer()

    def test_load_corpus(self):
        mock_corpus = MagicMock()
        with patch("convokit.Corpus", return_value=mock_corpus):
            self.chatbot_trainer.load_corpus("dummy_corpus_path")
        self.assertEqual(self.chatbot_trainer.corpus, mock_corpus)

    def test_setup_logger(self):
        logger = self.chatbot_trainer.setup_logger()
        self.assertIsNotNone(logger)
        self.assertEqual(len(logger.handlers), 2)  # Check if two handlers are added

    def test_preprocess_text(self):
        input_text = '   "Hello,    World!"   '
        cleaned_text = self.chatbot_trainer.preprocess_text(input_text)
        self.assertEqual(cleaned_text, "hello, world!")

    # You can add more test methods for other functionalities

if __name__ == "__main__":
    unittest.main()
