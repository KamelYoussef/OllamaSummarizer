import unittest
from src.functions import *


class TestTextProcessing(unittest.TestCase):

    def test_standardize_accented_chars(self):
        input_text = "résumé àçcéntéd"
        expected_output = "resume accented"
        self.assertEqual(standardize_accented_chars(input_text), expected_output)

    def test_remove_url(self):
        input_text = "Visit our website at https://www.example.com"
        expected_output = "Visit our website at "
        self.assertEqual(remove_url(input_text), expected_output)

    def test_expand_contractions(self):
        input_text = "I can't believe it's happening"
        expected_output = "I cannot believe it is happening"
        self.assertEqual(expand_contractions(input_text), expected_output)

    def test_remove_mentions_and_tags(self):
        input_text = "Check out this #awesome post by @user123"
        expected_output = "Check out this post by "
        self.assertEqual(remove_mentions_and_tags(input_text), expected_output)

    def test_remove_special_characters(self):
        input_text = "Remove *special* characters! @#$%^&"
        expected_output = "Remove special characters!"
        self.assertEqual(remove_special_characters(input_text), expected_output)

    def test_remove_spaces(self):
        input_text = "   Remove   extra  spaces     "
        expected_output = "Remove extra spaces"
        self.assertEqual(remove_spaces(input_text), expected_output)

    def test_remove_punctuation(self):
        input_text = "Remove all punctuation marks: ,.!?/:;\"'"
        expected_output = "Remove all punctuation marks "
        self.assertEqual(remove_punctuation(input_text), expected_output)


if __name__ == '__main__':
    unittest.main()
