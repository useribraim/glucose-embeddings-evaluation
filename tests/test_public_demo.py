"""Integrity failures must stop inference, including when reusing the cache."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.synthetic_demo import fetch_verified


class PublicDownloadChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        self.entry = {"path": "model/encoder.onnx", "url": "https://example.invalid/pinned-encoder",
                      "bytes": 5, "sha256": hashlib.sha256(b"valid").hexdigest()}

    def test_corrupt_cached_file_fails_without_network(self):
        target = self.cache / self.entry["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"wrong")
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "Cached file hash mismatch"):
                fetch_verified(self.entry, cache=self.cache)
            network.assert_not_called()

    def test_corrupt_download_is_not_installed(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "Downloaded file identity mismatch"):
                fetch_verified(self.entry, cache=self.cache)
        self.assertFalse((self.cache / self.entry["path"]).exists())

    def test_verified_cache_supports_offline_mode(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"valid")):
            path = fetch_verified(self.entry, cache=self.cache)
        with patch("urllib.request.urlopen") as network:
            self.assertEqual(fetch_verified(self.entry, cache=self.cache, offline=True), path)
            network.assert_not_called()

    def test_cache_escape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes the public cache"):
            fetch_verified(dict(self.entry, path="../escape"), cache=self.cache)


if __name__ == "__main__":
    unittest.main()
