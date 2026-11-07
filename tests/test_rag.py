from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import numpy as np

from rag import load_chunks, retrieve, classify, generate, embed, Chunk

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_real_policies_are_preserved_without_templates(self):
        chunks = load_chunks(ROOT / "data")
        self.assertEqual(len(chunks), 8)
        text = "\n".join(c.text for c in chunks)
        self.assertIn("1–2 business days", text)
        self.assertIn("5–7 business days", text)
        self.assertIn("within 30 days", text)
        self.assertIn("original condition and packaging", text)
        self.assertNotIn("processed your refund", text)
        self.assertNotIn("Template:", text)

    def test_routing_ranking_and_low_similarity(self):
        chunks = [Chunk("shipping", "s", 1, "tracking"), Chunk("refund", "r", 1, "refund"),
                  Chunk("shipping", "s", 2, "hours")]
        vectors = np.array([[0.8, 0.6], [1, 0], [0, 1]])
        query = np.array([1., 0.])
        hits = retrieve(chunks, vectors, query, "shipping")
        self.assertEqual([c.text for c, _ in hits], ["tracking"])
        self.assertEqual(retrieve(chunks, vectors, query, "mixed")[0][0].category, "refund")
        self.assertEqual(retrieve(chunks, vectors, query, "shipping", min_score=.9), [])
        self.assertEqual(retrieve(chunks, vectors, query, "other"), [])

    def test_embeddings_order_and_normalization(self):
        client = Mock()
        client.embeddings.create.return_value.data = [NS(index=1, embedding=[0, 2]), NS(index=0, embedding=[3, 0])]
        np.testing.assert_allclose(embed(client, "embedding", ["one", "two"]), np.eye(2))

    def test_classifier_validation_and_generation_context(self):
        client = Mock()
        client.chat.completions.create.return_value.choices = [NS(message=NS(content="refund"))]
        self.assertEqual(classify(client, "chat", "Return", "Please refund"), "refund")
        client.chat.completions.create.return_value.choices = [NS(message=NS(content="unknown"))]
        with self.assertRaises(ValueError):
            classify(client, "chat", "Return", "Please refund")
        client.chat.completions.create.return_value.choices = [NS(message=NS(content="Hi Alex"))]
        generate(client, "chat", "Alex", "Return", "Help", [(Chunk("refund", "r", 1, "30 days"), .9)])
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("30 days", messages[1]["content"])
        self.assertIn("NO access to orders", messages[0]["content"])


class InterfaceTests(unittest.TestCase):
    def test_no_key_and_required_fields(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py")).run()
        self.assertEqual(len(at.exception), 0)
        at.text_input(key="customer_name").set_value("Alex")
        at.text_input(key="email_subject").set_value("Where is my order?")
        at.text_area(key="email_body").set_value("My package is late. How can I track it?")
        at.button[0].click().run()
        self.assertIn("API key", at.error[0].value)
        next(w for w in at.text_input if w.label == "Customer name").set_value("")
        at.button[0].click().run()
        self.assertIn("customer name", at.error[0].value)

    def test_successful_form_shows_grounded_reply_and_evidence(self):
        from streamlit.testing.v1 import AppTest
        client = Mock()
        reply = "Hi Alex,\nPlease check your account for tracking.\nBest,\nThe Acme Support Team"
        client.chat.completions.create.side_effect = [
            NS(choices=[NS(message=NS(content="shipping"))]),
            NS(choices=[NS(message=NS(content=reply))]),
        ]
        client.embeddings.create.side_effect = [
            NS(data=[NS(index=i, embedding=[1., 0.]) for i in range(8)]),
            NS(data=[NS(index=0, embedding=[1., 0.])]),
        ]
        with patch("connection.create_client", return_value=client):
            at = AppTest.from_file(str(ROOT / "app.py")).run()
            at.text_input(key="customer_name").set_value("Alex")
            at.text_input(key="email_subject").set_value("Where is my order?")
            at.text_area(key="email_body").set_value("My package is late. How can I track it?")
            at.sidebar.text_input[0].set_value("test-key-not-real")
            at.button[0].click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.session_state["result"]["reply"], reply)
            self.assertEqual(len(at.session_state["result"]["hits"]), 3)
            self.assertEqual({c.category for c, _ in at.session_state["result"]["hits"]}, {"shipping"})
            self.assertEqual(len(at.session_state["history"]), 1)
            record = at.session_state["history"][0]
            self.assertNotIn("api_key", record)
            at.text_input(key="customer_name").set_value("Another customer")
            at.text_input(key="email_subject").set_value("Another subject")
            at.text_area(key="email_body").set_value("Another email").run()
            at.button(key="history_" + record["id"]).click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.text_input(key="customer_name").value, "Alex")
            self.assertEqual(at.text_input(key="email_subject").value, record["subject"])
            self.assertEqual(at.text_area(key="email_body").value, record["body"])
            self.assertEqual(at.session_state["result"]["reply"], reply)
            self.assertEqual(client.chat.completions.create.call_count, 2)


class ConnectionTests(unittest.TestCase):
    def test_chat_and_embedding_use_hkust_subscription_header(self):
        import httpx
        from connection import BASE_URL
        from openai import OpenAI
        from connection import create_client
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(200, json={"object": "list", "model": "text-embedding-3-small",
                    "data": [{"object": "embedding", "index": 0, "embedding": [1., 0.]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1}})
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "gpt-5.6-luna", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "shipping"}}]})

        # Run the real SDK locally; the mock transport prevents network access.
        def factory(**kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handle))
            return OpenAI(**kwargs)

        with patch("connection.OpenAI", side_effect=factory):
            with create_client(" test-key-not-real ") as client:
                client.chat.completions.create(model="gpt-5.6-luna", messages=[])
                client.embeddings.create(model="text-embedding-3-small", input=["policy"])
        self.assertEqual([r.url.path for r in requests],
                         ["/hkust-genai/v1/chat/completions", "/hkust-genai/v1/embeddings"])
        for request in requests:
            self.assertEqual(request.url.host, "hkust.azure-api.net")
            self.assertEqual(request.headers["api-key"], "test-key-not-real")


if __name__ == "__main__":
    unittest.main()
