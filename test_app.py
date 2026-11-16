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

    @patch("app.test_model")
    @patch("app.list_models")
    @patch("app.socket.getaddrinfo")
    def test_stream_check_yields_init_models_and_done(self, mock_addr, mock_list, mock_test):
        mock_addr.return_value = [(None, None, None, None, ("93.184.216.34", 443))]
        mock_list.return_value = ["model-a", "model-b"]
        mock_test.side_effect = [
            {"model": "model-a", "available": True, "latency_ms": 120.0, "response": {}},
            {"model": "model-b", "available": False, "latency_ms": None, "error": "timeout"},
        ]

        events = list(app.stream_check({"api_key": "key", "base_url": "https://provider.example/v1"}))
        self.assertEqual(events[0]["type"], "init")
        self.assertEqual(events[0]["total"], 2)

        model_events = [e for e in events if e["type"] == "model"]
        self.assertEqual(len(model_events), 2)
        self.assertEqual(model_events[0]["total"], 2)
        self.assertEqual(model_events[0]["tested"], 1)
        self.assertEqual(model_events[1]["tested"], 2)

        done_event = events[-1]
        self.assertEqual(done_event["type"], "done")
        self.assertEqual(done_event["data"]["total"], 2)
        self.assertEqual(done_event["data"]["available"], 1)
        self.assertEqual(done_event["data"]["ready"], 1)
        self.assertEqual(done_event["data"]["summary"]["median_ms"], 120.0)

    @patch("app.list_models")
    @patch("app.socket.getaddrinfo")
    def test_stream_check_cancellation(self, mock_addr, mock_list):
        import threading

        event = threading.Event()
        event.set()
        mock_addr.return_value = [(None, None, None, None, ("93.184.216.34", 443))]
        mock_list.return_value = ["model-a"]

        with self.assertRaisesRegex(app.ProviderError, "dibatalkan"):
            list(app.stream_check({"api_key": "key", "base_url": "https://provider.example/v1"}, event))

    @patch("app.stream_check")
    def test_handler_post_stream_writes_ndjson(self, mock_stream):
        import io
        import json
        from unittest.mock import MagicMock

        mock_stream.return_value = [
            {"type": "init", "total": 1},
            {"type": "done", "data": {"total": 1}},
        ]

        handler = app.Handler.__new__(app.Handler)
        handler.client_address = ("127.0.0.1", 12345)
        handler.path = "/api/check"
        payload = json.dumps({"stream": True, "api_key": "k", "base_url": "https://example.com/v1"}).encode("utf-8")
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        with patch("app.validate_check_config"):
            handler.do_POST()

        output = handler.wfile.getvalue().decode("utf-8")
        lines = [json.loads(line) for line in output.strip().split("\n")]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["type"], "init")
        self.assertEqual(lines[1]["type"], "done")


if __name__ == "__main__":
    unittest.main()
