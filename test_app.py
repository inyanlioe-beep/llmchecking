import unittest
from unittest.mock import patch

import app


class AppTests(unittest.TestCase):
    @patch("app.request_json")
    def test_list_models_filters_case_insensitively_and_caps_count(self, request_json):
        request_json.return_value = ({"data": [{"id": "gpt-1"}, {"id": "claude-1"}, {"id": "GPT-2"}]}, 1)

        self.assertEqual(app.list_models("https://provider.example/v1", "key", "GpT", 1), ["GPT-2"])

    def test_private_network_base_url_is_rejected(self):
        with patch("app.socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 443))]):
            with self.assertRaisesRegex(app.ProviderError, "alamat jaringan privat"):
                app.run_check({"api_key": "key", "base_url": "https://provider.example/v1"})

    def test_cancellation_before_testing_stops_run(self):
        import threading

        event = threading.Event()
        event.set()
        with patch("app.socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 443))]), patch("app.list_models", return_value=["model"]):
            with self.assertRaisesRegex(app.ProviderError, "dibatalkan"):
                app.run_check({"api_key": "key", "base_url": "https://provider.example/v1"}, event)


if __name__ == "__main__":
    unittest.main()
