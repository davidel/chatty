import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from chatty.auth_proxy import (
  AuthProvider,
  BearerTokenProvider,
  BoundedStream,
  CredentialProxy,
  GCSAuthProvider,
  HeaderAuthProvider,
  ProxyRedirectHandler,
  ProxyRoute,
  RWUrlFilter,
  SigV4AuthProvider,
  create_default_credential_proxy,
  is_allowed_client_ip
)


class TestIPVerification(unittest.TestCase):

  def test_allowed_ips(self):
    self.assertTrue(is_allowed_client_ip("127.0.0.1"))
    self.assertTrue(is_allowed_client_ip("::1"))
    self.assertTrue(is_allowed_client_ip("172.17.0.2"))
    self.assertTrue(is_allowed_client_ip("10.0.4.5"))
    self.assertTrue(is_allowed_client_ip("192.168.1.100"))

  def test_disallowed_ips(self):
    self.assertFalse(is_allowed_client_ip("8.8.8.8"))
    self.assertFalse(is_allowed_client_ip("1.1.1.1"))
    self.assertFalse(is_allowed_client_ip("invalid_ip"))


class TestAuthProviders(unittest.TestCase):

  def test_bearer_token_provider(self):
    provider = BearerTokenProvider("hf_secret_token_12345")
    headers = provider.get_headers("GET", "https://huggingface.co/api/models", None)
    self.assertEqual(headers, {"Authorization": "Bearer hf_secret_token_12345"})
    self.assertEqual(provider.get_secrets(), ["hf_secret_token_12345"])

  def test_bearer_token_provider_from_env(self):
    with patch.dict(os.environ, {"CUSTOM_HF_TOKEN": "hf_env_token_abcde"}):
      provider = BearerTokenProvider("CUSTOM_HF_TOKEN")
      headers = provider.get_headers("GET", "https://huggingface.co/api/models", None)
      self.assertEqual(headers, {"Authorization": "Bearer hf_env_token_abcde"})
      self.assertEqual(provider.get_secrets(), ["hf_env_token_abcde"])

  def test_header_auth_provider(self):
    provider = HeaderAuthProvider(
      headers=lambda: {"x-api-key": "anthropic-secret-key-123", "anthropic-version": "2023-06-01"},
      secrets=lambda: ["anthropic-secret-key-123"]
    )
    headers = provider.get_headers("POST", "https://api.anthropic.com/v1/messages", None)
    self.assertEqual(headers["x-api-key"], "anthropic-secret-key-123")
    self.assertEqual(headers["anthropic-version"], "2023-06-01")
    self.assertEqual(provider.get_secrets(), ["anthropic-secret-key-123"])

  def test_sigv4_auth_provider(self):
    provider = SigV4AuthProvider(
      access_key="AKIAEXAMPLEKEY123",
      secret_key="SKIAEXAMPLESECRETKEY456",
      service="s3",
      region="us-east-1"
    )
    headers = provider.get_headers("GET", "https://s3.amazonaws.com/my-bucket/file.txt", None)
    self.assertIn("Host", headers)
    self.assertIn("Authorization", headers)
    self.assertTrue(headers["Authorization"].startswith("AWS4-HMAC-SHA256"))
    self.assertIn("AKIAEXAMPLEKEY123", headers["Authorization"])
    secrets = provider.get_secrets()
    self.assertIn("AKIAEXAMPLEKEY123", secrets)
    self.assertIn("SKIAEXAMPLESECRETKEY456", secrets)

  def test_gcs_auth_provider(self):
    provider = GCSAuthProvider()
    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds.token = "ya29.mock_oauth2_token_xyz"
    provider._credentials = mock_creds

    headers = provider.get_headers("GET", "https://storage.googleapis.com/b/bucket/o", None)
    self.assertEqual(headers, {"Authorization": "Bearer ya29.mock_oauth2_token_xyz"})
    self.assertEqual(provider.get_secrets(), ["ya29.mock_oauth2_token_xyz"])


class TestCredentialProxyRouting(unittest.TestCase):

  def test_route_matching_and_env(self):
    proxy = CredentialProxy()
    route_hf = ProxyRoute(
      prefix="/hf",
      target_base_url="https://huggingface.co",
      auth_provider=BearerTokenProvider("hf_test_token"),
      container_env_var="HF_ENDPOINT",
      allowed_methods=frozenset({"GET", "HEAD"})
    )
    route_openai = ProxyRoute(
      prefix="/openai",
      target_base_url="https://api.openai.com",
      auth_provider=BearerTokenProvider("sk_test_key"),
      container_env_var="OPENAI_BASE_URL",
      allowed_methods=frozenset({"GET", "POST"})
    )
    proxy.add_route(route_hf)
    proxy.add_route(route_openai)

    # Test route matching
    r, subpath = proxy.match_route("/hf/api/models/test")
    self.assertIs(r, route_hf)
    self.assertEqual(subpath, "/api/models/test")

    r, subpath = proxy.match_route("/openai/v1/chat/completions")
    self.assertIs(r, route_openai)
    self.assertEqual(subpath, "/v1/chat/completions")

    r, subpath = proxy.match_route("/unknown/path")
    self.assertIsNone(r)

    # Test redaction patterns
    patterns = proxy.get_redaction_patterns()
    self.assertIn("hf_test_token", patterns)
    self.assertIn("sk_test_key", patterns)

  def test_create_default_credential_proxy(self):
    test_env = {
      "HF_TOKEN": "hf_test_token_999",
      "OPENAI_API_KEY": "sk_test_openai_key",
      "CLOUDFLARE_ACCOUNT_ID": "cf_account_123",
      "R2_ACCESS_KEY_ID": "r2_access_key",
      "R2_SECRET_ACCESS_KEY": "r2_secret_key",
      "GITHUB_TOKEN": "ghp_github_token_xyz"
    }
    proxy = create_default_credential_proxy(env=test_env)
    prefixes = [r.prefix for r in proxy.routes]
    self.assertIn("/hf", prefixes)
    self.assertIn("/openai", prefixes)
    self.assertIn("/r2", prefixes)
    self.assertIn("/github", prefixes)


class TestCredentialProxyServer(unittest.TestCase):

  def setUp(self):
    self.proxy = CredentialProxy()
    self.route = ProxyRoute(
      prefix="/hf",
      target_base_url="https://huggingface.co",
      auth_provider=BearerTokenProvider("hf_mock_secret_token"),
      container_env_var="HF_ENDPOINT",
      allowed_methods=frozenset({"GET", "HEAD"})
    )
    self.proxy.add_route(self.route)
    self.proxy.start()

  def tearDown(self):
    self.proxy.stop()

  def test_proxy_lifecycle(self):
    self.assertTrue(self.proxy.is_running)
    self.assertGreater(self.proxy.port, 0)
    container_env = self.proxy.get_container_env("host.docker.internal")
    self.assertIn("HF_ENDPOINT", container_env)
    self.assertEqual(container_env["HF_ENDPOINT"], f"http://host.docker.internal:{self.proxy.port}/hf")

  @patch("chatty.auth_proxy.proxy_opener.open")
  def test_successful_request_dispatch(self, mock_urlopen):
    import http.client
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getheaders.return_value = [("Content-Type", "application/json")]
    mock_resp.read.side_effect = [b'{"status": "ok"}', b""]
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    conn = http.client.HTTPConnection("127.0.0.1", self.proxy.port, timeout=5)
    conn.request("GET", "/hf/api/whoami")
    response = conn.getresponse()
    self.assertEqual(response.status, 200)
    data = response.read()
    self.assertEqual(data, b'{"status": "ok"}')
    conn.close()

    # Verify upstream received authorization header
    mock_urlopen.assert_called_once()
    upstream_req = mock_urlopen.call_args[0][0]
    self.assertEqual(upstream_req.full_url, "https://huggingface.co/api/whoami")
    self.assertEqual(upstream_req.headers.get("Authorization"), "Bearer hf_mock_secret_token")
    self.assertEqual(upstream_req.host, "huggingface.co")

  def test_method_not_allowed(self):
    proxy_url = f"http://127.0.0.1:{self.proxy.port}/hf/api/delete_something"
    req = urllib.request.Request(proxy_url, data=b"{}", method="POST")
    with self.assertRaises(urllib.error.HTTPError) as ctx:
      urllib.request.urlopen(req)
    self.assertEqual(ctx.exception.code, 403)

  def test_route_not_found(self):
    proxy_url = f"http://127.0.0.1:{self.proxy.port}/nonexistent/endpoint"
    req = urllib.request.Request(proxy_url, method="GET")
    with self.assertRaises(urllib.error.HTTPError) as ctx:
      urllib.request.urlopen(req)
    self.assertEqual(ctx.exception.code, 404)


class TestSecretScrubbing(unittest.TestCase):

  def test_scrub_secrets_in_runner(self):
    from chatty.session import ChatbotSession, SessionConfig
    config = SessionConfig(sandbox="/tmp", provider="dummy", model="dummy")
    session = ChatbotSession(sandbox="/tmp", config=config)
    runner = session.runner

    with patch.dict(os.environ, {"HF_TOKEN": "hf_super_secret_token_123"}):
      dirty_text = "Error downloading model with token hf_super_secret_token_123: 403 Forbidden"
      cleaned = runner._scrub_secrets(dirty_text)
      self.assertNotIn("hf_super_secret_token_123", cleaned)
      self.assertIn("[REDACTED_SECRET]", cleaned)


class TestBoundedStream(unittest.TestCase):

  def test_stream_reads_correctly(self):
    raw = io.BytesIO(b"Hello Antigravity Cloud Storage")
    stream = BoundedStream(raw, 17)
    self.assertTrue(stream.readable())
    self.assertFalse(stream.seekable())
    self.assertFalse(stream.writable())

    first = stream.read(5)
    self.assertEqual(first, b"Hello")
    rest = stream.read()
    self.assertEqual(rest, b" Antigravity")
    empty = stream.read(10)
    self.assertEqual(empty, b"")


class TestRWUrlFilter(unittest.TestCase):

  def test_empty_filter(self):
    rw = RWUrlFilter(None)
    self.assertFalse(rw.is_write_allowed("/s3", "/my-bucket/key"))
    self.assertFalse(rw.has_gcs_rules())

  def test_matching_s3_and_r2(self):
    rw = RWUrlFilter("s3://scratch-bucket;r2://artifacts/*;eval-bucket")
    self.assertTrue(rw.is_write_allowed("/s3", "/scratch-bucket/model.bin"))
    self.assertTrue(rw.is_write_allowed("/r2", "/artifacts/checkpoint-100/weights.pt"))
    self.assertFalse(rw.is_write_allowed("/r2", "/other/checkpoint-100/weights.pt"))
    self.assertTrue(rw.is_write_allowed("/s3", "/eval-bucket/report.json"))
    self.assertFalse(rw.is_write_allowed("/s3", "/prod-bucket/secrets.env"))

  def test_gcs_detection_and_matching(self):
    rw = RWUrlFilter("gs://gcs-staging/*")
    self.assertTrue(rw.has_gcs_rules())
    self.assertTrue(rw.is_write_allowed("/gcs", "/b/gcs-staging/o/data.csv"))
    self.assertTrue(rw.is_write_allowed("/gcs", "/gcs-staging/data.csv"))
    self.assertFalse(rw.is_write_allowed("/gcs", "/b/gcs-prod/o/data.csv"))


class TestSelectiveRWProxy(unittest.TestCase):

  def setUp(self):
    self.rw_filter = RWUrlFilter("s3://allowed-bucket;allowed-bucket/*")
    self.proxy = CredentialProxy(rw_filter=self.rw_filter)
    self.s3_route = ProxyRoute(
      prefix="/s3",
      target_base_url="https://s3.amazonaws.com",
      auth_provider=SigV4AuthProvider(
        access_key="AKIA123",
        secret_key="SKIA456",
        service="s3",
        region="us-east-1"
      ),
      allowed_methods=frozenset({"GET", "HEAD", "PUT", "POST", "DELETE"}),
      rw_filter=self.rw_filter
    )
    self.proxy.add_route(self.s3_route)
    self.proxy.start()

  def tearDown(self):
    self.proxy.stop()

  @patch("chatty.auth_proxy.proxy_opener.open")
  def test_put_allowed_bucket_succeeds(self, mock_urlopen):
    import http.client
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getheaders.return_value = [("ETag", '"abc123etag"')]
    mock_resp.read.side_effect = [b"", b""]
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    conn = http.client.HTTPConnection("127.0.0.1", self.proxy.port, timeout=5)
    conn.request("PUT", "/s3/allowed-bucket/test.txt", body=b"hello s3 content")
    resp = conn.getresponse()
    self.assertEqual(resp.status, 200)
    conn.close()

    mock_urlopen.assert_called_once()
    upstream_req = mock_urlopen.call_args[0][0]
    self.assertEqual(upstream_req.headers.get("X-amz-content-sha256"), "UNSIGNED-PAYLOAD")
    self.assertIn("Authorization", upstream_req.headers)

  def test_put_forbidden_bucket_returns_403(self):
    proxy_url = f"http://127.0.0.1:{self.proxy.port}/s3/forbidden-bucket/test.txt"
    req = urllib.request.Request(proxy_url, data=b"malicious write", method="PUT")
    with self.assertRaises(urllib.error.HTTPError) as ctx:
      urllib.request.urlopen(req)
    self.assertEqual(ctx.exception.code, 403)
    self.assertIn("CHATTY_RW_URLS", ctx.exception.reason)

  @patch("chatty.auth_proxy.proxy_opener.open")
  def test_get_allowed_on_any_bucket(self, mock_urlopen):
    import http.client
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getheaders.return_value = [("Content-Type", "text/plain")]
    mock_resp.read.side_effect = [b"read data", b""]
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    conn = http.client.HTTPConnection("127.0.0.1", self.proxy.port, timeout=5)
    conn.request("GET", "/s3/any-other-bucket/test.txt")
    resp = conn.getresponse()
    self.assertEqual(resp.status, 200)
    data = resp.read()
    self.assertEqual(data, b"read data")
    conn.close()


class TestProxyRedirectHandler(unittest.TestCase):

  def setUp(self):
    self.handler = ProxyRedirectHandler()

  def test_same_domain_redirect_followed(self):
    req = urllib.request.Request("https://huggingface.co/api/datasets/test")
    new_req = self.handler.redirect_request(
      req, None, 307, "Temporary Redirect", {}, "https://huggingface.co/api/resolve-cache/test"
    )
    self.assertIsNotNone(new_req)
    self.assertEqual(new_req.full_url, "https://huggingface.co/api/resolve-cache/test")

  def test_relative_redirect_followed(self):
    req = urllib.request.Request("https://huggingface.co/api/datasets/test")
    new_req = self.handler.redirect_request(
      req, None, 307, "Temporary Redirect", {}, "/api/resolve-cache/relative-test"
    )
    self.assertIsNotNone(new_req)
    self.assertIn("/api/resolve-cache/relative-test", new_req.full_url)

  def test_cross_domain_cdn_redirect_stopped(self):
    req = urllib.request.Request("https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/data.parquet")
    new_req = self.handler.redirect_request(
      req, None, 302, "Found", {}, "https://us.aws.cdn.hf.co/xet-bridge/data.parquet"
    )
    self.assertIsNone(new_req)
