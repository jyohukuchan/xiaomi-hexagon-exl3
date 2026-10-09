from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from api_probe import Client, check_completion, check_usage


class ApiValidationTest(unittest.TestCase):
    def response(self, chat):
        choice = {"index": 0, "finish_reason": "stop"}
        choice.update({"message": {"role": "assistant", "content": "The cat is sleeping."}} if chat else {"text": "The cat is sleeping."})
        return {"id": "local-test", "created": 1, "model": "index-translate-2b-exl3",
                "object": "chat.completion" if chat else "text_completion", "choices": [choice],
                "usage": {"prompt_tokens": 21, "completion_tokens": 6, "total_tokens": 27}}

    def test_completion_schemas(self):
        for chat in (False, True):
            self.assertEqual(check_completion(self.response(chat), chat), "The cat is sleeping.")

    def test_usage_requires_integer_counts(self):
        for invalid in (-1, 1.0, True, None):
            with self.assertRaises(AssertionError):
                check_usage({"prompt_tokens": invalid, "completion_tokens": 6, "total_tokens": 27})

    def test_usage_requires_sum(self):
        with self.assertRaises(AssertionError):
            check_usage({"prompt_tokens": 21, "completion_tokens": 6, "total_tokens": 28})

    def test_incorrect_role_rejected(self):
        value = self.response(True)
        value["choices"][0]["message"]["role"] = "user"
        with self.assertRaises(AssertionError):
            check_completion(value, True)

    def test_incorrect_finish_reason_rejected(self):
        value = self.response(False)
        value["choices"][0]["finish_reason"] = None
        with self.assertRaises(AssertionError):
            check_completion(value, False)

    def test_external_endpoints_rejected(self):
        for url in ("https://api.openai.com/v1", "http://192.168.0.1", "https://127.0.0.1", "http://user:secret@localhost", "http://localhost/?key=secret"):
            with self.assertRaises(ValueError):
                Client(url)
        Client("http://127.0.0.1:8088")


if __name__ == "__main__":
    unittest.main()
