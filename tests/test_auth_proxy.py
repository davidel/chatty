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
  CredentialProxy,
  GCSAuthProvider,
  HeaderAuthProvider,
  ProxyRoute,
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

  @patch("chatty.auth_proxy.urllib.request.urlopen")
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
    self.assertEqual(upstream_req.headers.get("Host"), "huggingface.co")

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
