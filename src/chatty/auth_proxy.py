"""Credential-injecting egress proxy for sandboxed Docker containers.

Safely injects authentication credentials (HF_TOKEN, AWS SigV4, GCS OAuth2, etc.)
for outbound API calls originating from sandboxed commands without exposing the
underlying secrets to the LLM or container filesystem.
"""

import abc
import contextlib
import fnmatch
import io
import ipaddress
import logging
import os
import shutil
import socketserver
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger("chatty")

HOP_BY_HOP_HEADERS = frozenset({
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailers",
  "transfer-encoding",
  "upgrade",
  "content-length"
})


def is_allowed_client_ip(client_ip: str) -> bool:
  """Verifies that an incoming request originates from localhost or a private Docker bridge network."""
  try:
    ip = ipaddress.ip_address(client_ip)
    return ip.is_loopback or ip.is_private
  except Exception:
    return False


class BoundedStream(io.RawIOBase):
  """Streams request body from client socket bounded by Content-Length without buffering into RAM."""

  def __init__(self, raw_socket, length: int):
    self.raw = raw_socket
    self.remaining = length

  def read(self, size: int = -1) -> bytes:
    if self.remaining <= 0:
      return b""
    to_read = self.remaining if size < 0 else min(size, self.remaining)
    chunk = self.raw.read(to_read)
    if not chunk:
      self.remaining = 0
      return b""
    self.remaining -= len(chunk)
    return chunk

  def readable(self) -> bool:
    return True

  def seekable(self) -> bool:
    return False

  def writable(self) -> bool:
    return False


class RWUrlFilter:
  """Determines whether write operations (PUT, POST, DELETE, PATCH) are permitted for a route and subpath."""

  def __init__(self, raw_rules: Optional[str] = None):
    self.raw_rules = (raw_rules or "").strip()
    self.patterns: List[str] = []
    if self.raw_rules:
      for part in self.raw_rules.split(";"):
        p = part.strip()
        if p:
          self.patterns.append(p)

  def has_gcs_rules(self) -> bool:
    """Returns True if any rule targets or might target Google Cloud Storage."""
    if not self.patterns:
      return False
    for pat in self.patterns:
      pat_lower = pat.lower()
      if pat == "*" or "gcs" in pat_lower or "gs://" in pat_lower or "google" in pat_lower:
        return True
      if "://" not in pat:
        return True
    return False

  def is_write_allowed(self, prefix: str, subpath: str, target_base_url: str = "") -> bool:
    """Checks if writing to prefix + subpath is allowed by any whitelist rule."""
    if not self.patterns:
      return False

    svc = prefix.lstrip("/").lower()
    clean_subpath = subpath.lstrip("/")

    bucket_name = ""
    obj_path = ""
    if "/b/" in subpath:
      parts = subpath.split("/b/")
      if len(parts) > 1:
        bucket_and_rest = parts[1].split("?")[0]
        if "/o/" in bucket_and_rest:
          b_parts = bucket_and_rest.split("/o/")
          bucket_name = b_parts[0]
          obj_path = b_parts[1]
        else:
          bucket_name = bucket_and_rest.split("/")[0]
    elif clean_subpath:
      sub_parts = clean_subpath.split("?")[0].split("/", 1)
      bucket_name = sub_parts[0]
      if len(sub_parts) > 1:
        obj_path = sub_parts[1]

    candidates: Set[str] = {
      clean_subpath,
      f"/{svc}/{clean_subpath}",
      f"{svc}://{clean_subpath}",
    }
    if bucket_name:
      candidates.add(bucket_name)
      candidates.add(f"{svc}://{bucket_name}")
      if svc in ("s3", "r2"):
        candidates.add(f"s3://{bucket_name}")
        candidates.add(f"r2://{bucket_name}")
      if svc == "gcs":
        candidates.add(f"gs://{bucket_name}")
      if obj_path:
        candidates.add(f"{bucket_name}/{obj_path}")
        candidates.add(f"{svc}://{bucket_name}/{obj_path}")
        if svc in ("s3", "r2"):
          candidates.add(f"s3://{bucket_name}/{obj_path}")
          candidates.add(f"r2://{bucket_name}/{obj_path}")
        if svc == "gcs":
          candidates.add(f"gs://{bucket_name}/{obj_path}")
    if target_base_url:
      target_clean = target_base_url.rstrip("/")
      candidates.add(f"{target_clean}/{clean_subpath}")

    for pattern in self.patterns:
      pat = pattern.strip()
      pat_base = pat[:-2] if pat.endswith("/*") else pat
      for cand in candidates:
        if (
          fnmatch.fnmatch(cand, pat)
          or fnmatch.fnmatch(cand, f"{pat}/*")
          or fnmatch.fnmatch(cand, f"{pat}*")
          or fnmatch.fnmatch(cand, pat_base)
        ):
          return True

    return False


class AuthProvider(abc.ABC):
  """Abstract base class for credential injectors."""

  @abc.abstractmethod
  def get_headers(self, method: str, url: str, body: Optional[Any] = None) -> Dict[str, str]:
    """Generates the required authentication headers for the upstream request."""
    pass

  def clean_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
    """Sanitizes incoming client headers before injecting credentials."""
    return dict(headers)

  def get_secrets(self) -> List[str]:
    """Returns secret strings associated with this provider for output scrubbing."""
    return []


class BearerTokenProvider(AuthProvider):
  """Injects standard Bearer tokens (e.g. HF_TOKEN, OPENAI_API_KEY, GITHUB_TOKEN)."""

  def __init__(self, token_or_env_var: str):
    self._token_or_env_var = token_or_env_var

  def _resolve_token(self) -> str:
    val = os.environ.get(self._token_or_env_var, self._token_or_env_var).strip()
    return val

  def clean_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() != "authorization"}

  def get_headers(self, method: str, url: str, body: Optional[Any] = None) -> Dict[str, str]:
    token = self._resolve_token()
    return {"Authorization": f"Bearer {token}"} if token else {}

  def get_secrets(self) -> List[str]:
    token = self._resolve_token()
    return [token] if token and len(token) >= 4 else []


class HeaderAuthProvider(AuthProvider):
  """Injects arbitrary custom headers via a callable."""

  def __init__(
    self,
    headers: Callable[[], Dict[str, str]],
    secrets: Optional[Callable[[], List[str]]] = None
  ):
    self._headers_fn = headers
    self._secrets_fn = secrets

  def get_headers(self, method: str, url: str, body: Optional[Any] = None) -> Dict[str, str]:
    try:
      return dict(self._headers_fn())
    except Exception as e:
      logger.warning(f"Error computing auth headers: {e}")
      return {}

  def get_secrets(self) -> List[str]:
    if self._secrets_fn:
      try:
        return [s for s in self._secrets_fn() if s and len(s) >= 4]
      except Exception:
        return []
    return []


class SigV4AuthProvider(AuthProvider):
  """Signs S3/R2 requests dynamically using AWS Signature Version 4 (SigV4)."""

  SIGV4_STALE_HEADERS = frozenset({
    "authorization",
    "x-amz-date",
    "x-amz-security-token",
    "x-amz-content-sha256"
  })

  def __init__(
    self,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    service: str = "s3",
    region: str = "auto",
    use_unsigned_payload: bool = True
  ):
    self._access_key = access_key
    self._secret_key = secret_key
    self.service = service
    self.region = region
    self.use_unsigned_payload = use_unsigned_payload

  def _get_credentials(self) -> Tuple[str, str]:
    ak = self._access_key or os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID") or ""
    sk = self._secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY") or ""
    return ak.strip(), sk.strip()

  def clean_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in self.SIGV4_STALE_HEADERS}

  def get_headers(self, method: str, url: str, body: Optional[Any] = None) -> Dict[str, str]:
    ak, sk = self._get_credentials()
    if not ak or not sk:
      return {}
    try:
      import botocore.auth
      import botocore.awsrequest
      import botocore.credentials

      creds = botocore.credentials.Credentials(ak, sk)
      signer = botocore.auth.SigV4Auth(creds, self.service, self.region)
      parsed = urllib.parse.urlparse(url)
      headers = {"Host": parsed.netloc}

      is_write_method = method.upper() in {"PUT", "POST", "PATCH"}
      if self.use_unsigned_payload and (body is not None or is_write_method):
        headers["x-amz-content-sha256"] = "UNSIGNED-PAYLOAD"
        sign_data = None
      else:
        sign_data = body if isinstance(body, (bytes, bytearray)) else None

      aws_request = botocore.awsrequest.AWSRequest(
        method=method,
        url=url,
        headers=headers,
        data=sign_data
      )
      signer.add_auth(aws_request)
      return dict(aws_request.headers)
    except Exception as e:
      logger.warning(f"Failed to sign request with AWS SigV4: {e}")
      return {}

  def get_secrets(self) -> List[str]:
    ak, sk = self._get_credentials()
    secrets = []
    if ak and len(ak) >= 4:
      secrets.append(ak)
    if sk and len(sk) >= 4:
      secrets.append(sk)
    return secrets


class GCSAuthProvider(AuthProvider):
  """Refreshes and injects Google Cloud OAuth2 Access Tokens on the host."""

  def __init__(self, scopes: Optional[List[str]] = None):
    self.scopes = scopes or ["https://www.googleapis.com/auth/devstorage.read_only"]
    self._credentials = None
    self._auth_request = None
    self._lock = threading.Lock()

  def _ensure_credentials(self):
    if self._credentials is None:
      import google.auth
      import google.auth.transport.requests
      self._credentials, _ = google.auth.default(scopes=self.scopes)
      self._auth_request = google.auth.transport.requests.Request()

  def get_headers(self, method: str, url: str, body: Optional[bytes]) -> Dict[str, str]:
    try:
      with self._lock:
        self._ensure_credentials()
        if not self._credentials.valid:
          self._credentials.refresh(self._auth_request)
        token = self._credentials.token
      return {"Authorization": f"Bearer {token}"}
    except Exception as e:
      logger.warning(f"Failed to obtain GCS OAuth2 token: {e}")
      return {}

  def get_secrets(self) -> List[str]:
    if self._credentials and getattr(self._credentials, "token", None):
      tok = str(self._credentials.token)
      if len(tok) >= 4:
        return [tok]
    return []


class ProxyRoute:
  """Defines an upstream route prefix, its target URL, and allowed operations."""

  def __init__(
    self,
    prefix: str,
    target_base_url: str,
    auth_provider: AuthProvider,
    container_env_var: Optional[str] = None,
    allowed_methods: FrozenSet[str] = frozenset({"GET", "HEAD"}),
    path_rewriter: Optional[Callable[[str], str]] = None,
    path_filter: Optional[Callable[[str], bool]] = None,
    rw_filter: Optional[RWUrlFilter] = None
  ):
    self.prefix = prefix.rstrip("/")
    self.target_base_url = target_base_url.rstrip("/")
    self.auth_provider = auth_provider
    self.container_env_var = container_env_var
    self.allowed_methods = allowed_methods
    self.path_rewriter = path_rewriter
    self.path_filter = path_filter
    self.rw_filter = rw_filter


class ThreadingAuthProxyServer(socketserver.ThreadingMixIn, HTTPServer):
  daemon_threads = True
  allow_reuse_address = True


class ProxyRedirectHandler(urllib.request.HTTPRedirectHandler):
  """Custom redirect handler for the credential proxy.

  Follows internal redirects on the same domain (e.g. resolve-cache),
  but stops at cross-domain redirects (e.g. presigned S3 / CloudFront CDN)
  so that critical headers (like X-Repo-Commit, Location) are preserved and returned
  to the sandboxed client.
  """

  def redirect_request(self, req, fp, code, msg, headers, newurl):
    newurl = urllib.parse.urljoin(req.full_url, newurl)
    parsed_new = urllib.parse.urlparse(newurl)
    parsed_orig = urllib.parse.urlparse(req.full_url)
    if parsed_new.netloc.lower() == parsed_orig.netloc.lower():
      new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
      if new_req and "Host" in new_req.headers:
        del new_req.headers["Host"]
      return new_req
    return None


proxy_opener = urllib.request.build_opener(ProxyRedirectHandler)


class AuthProxyHandler(BaseHTTPRequestHandler):
  """HTTP request handler for dispatching requests to upstream services."""

  server: ThreadingAuthProxyServer

  def do_GET(self):
    self._handle_request("GET")

  def do_HEAD(self):
    self._handle_request("HEAD")

  def do_POST(self):
    self._handle_request("POST")

  def do_PUT(self):
    self._handle_request("PUT")

  def do_DELETE(self):
    self._handle_request("DELETE")

  def do_OPTIONS(self):
    self._handle_request("OPTIONS")

  def _handle_request(self, method: str):
    client_ip = self.client_address[0]
    if not is_allowed_client_ip(client_ip):
      self.send_error(403, "Access denied: client IP not allowed.")
      return

    proxy_instance: "CredentialProxy" = getattr(self.server, "credential_proxy", None)
    if not proxy_instance:
      self.send_error(500, "Credential proxy instance missing on server.")
      return

    route, subpath = proxy_instance.match_route(self.path)
    if not route:
      self.send_error(404, f"Route prefix for '{self.path}' not found on credential proxy.")
      return

    if method not in route.allowed_methods:
      self.send_error(403, f"Method '{method}' is not permitted for '{route.prefix}'.")
      return

    if method in {"PUT", "POST", "DELETE", "PATCH"} and route.rw_filter is not None:
      if not route.rw_filter.is_write_allowed(route.prefix, subpath, route.target_base_url):
        self.send_error(
          403,
          f"Write access ('{method}') to '{route.prefix}{subpath}' denied by policy. "
          f"Target bucket or URL is not permitted by CHATTY_RW_URLS."
        )
        return

    if route.path_filter and not route.path_filter(subpath):
      self.send_error(403, f"Access to '{subpath}' is prohibited by policy.")
      return

    if route.path_rewriter:
      subpath = route.path_rewriter(subpath)

    target_url = f"{route.target_base_url}{subpath}"
    headers = {
      k: v for k, v in self.headers.items()
      if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() != "host"
    }
    headers = route.auth_provider.clean_headers(headers)

    content_len = int(self.headers.get("Content-Length", 0))
    body_stream = BoundedStream(self.rfile, content_len) if content_len > 0 else None

    # Inject authenticated headers dynamically on the host
    auth_headers = route.auth_provider.get_headers(method, target_url, body_stream)
    headers.update(auth_headers)

    req = urllib.request.Request(target_url, data=body_stream, headers=headers, method=method)
    try:
      with proxy_opener.open(req, timeout=120) as resp:
        self.send_response(resp.status)
        for k, v in resp.getheaders():
          if k.lower() not in HOP_BY_HOP_HEADERS:
            self.send_header(k, v)
        self.end_headers()
        shutil.copyfileobj(resp, self.wfile, length=64 * 1024)
    except urllib.error.HTTPError as e:
      self.send_response(e.code)
      for k, v in e.headers.items():
        if k.lower() not in HOP_BY_HOP_HEADERS:
          self.send_header(k, v)
      self.end_headers()
      shutil.copyfileobj(e, self.wfile, length=64 * 1024)
    except urllib.error.URLError as e:
      self.send_error(502, f"Bad Gateway: {e.reason}")
    except Exception as e:
      self.send_error(500, f"Proxy internal error: {e}")

  def log_message(self, format, *args):
    """Silences standard HTTP server request logs from polluting console."""
    pass


class CredentialProxy:
  """Manages the background HTTP proxy server lifecycle and registered routes."""

  def __init__(
    self,
    routes: Optional[List[ProxyRoute]] = None,
    bind_host: str = "0.0.0.0",
    rw_filter: Optional[RWUrlFilter] = None
  ):
    self.routes: List[ProxyRoute] = list(routes) if routes else []
    self.bind_host = bind_host
    self.rw_filter = rw_filter
    self.port: int = 0
    self._server: Optional[ThreadingAuthProxyServer] = None
    self._thread: Optional[threading.Thread] = None
    self.is_running: bool = False
    self.port: int = 0
    self._server: Optional[ThreadingAuthProxyServer] = None
    self._thread: Optional[threading.Thread] = None
    self.is_running: bool = False

  def add_route(self, route: ProxyRoute) -> None:
    self.routes.append(route)

  def match_route(self, path: str) -> Tuple[Optional[ProxyRoute], str]:
    sorted_routes = sorted(self.routes, key=lambda r: len(r.prefix), reverse=True)
    for route in sorted_routes:
      if path == route.prefix or path.startswith(route.prefix + "/") or path.startswith(route.prefix + "?"):
        subpath = path[len(route.prefix):]
        if not subpath.startswith("/") and not subpath.startswith("?"):
          subpath = "/" + subpath
        return route, subpath
    return None, path

  def start(self) -> None:
    if self.is_running:
      return
    self._server = ThreadingAuthProxyServer((self.bind_host, 0), AuthProxyHandler)
    self._server.credential_proxy = self
    self.port = self._server.server_address[1]
    self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
    self._thread.start()
    self.is_running = True
    logger.info(f"CredentialProxy started on {self.bind_host}:{self.port} with {len(self.routes)} route(s).")

  def stop(self) -> None:
    if not self.is_running or not self._server:
      return
    try:
      self._server.shutdown()
      self._server.server_close()
    except Exception as e:
      logger.debug(f"Error stopping CredentialProxy: {e}")
    finally:
      self._server = None
      self._thread = None
      self.is_running = False
      logger.info("CredentialProxy stopped.")

  def get_container_env(self, host_name: str = "host.docker.internal") -> Dict[str, str]:
    """Generates the environment variables to pass into the companion Docker container."""
    env = {}
    base_url = f"http://{host_name}:{self.port}"
    for route in self.routes:
      if route.container_env_var:
        env[route.container_env_var] = f"{base_url}{route.prefix}"
    return env

  def get_redaction_patterns(self) -> Set[str]:
    """Collects all configured secret values to scrub from stdout/stderr output."""
    patterns: Set[str] = set()
    for route in self.routes:
      for secret in route.auth_provider.get_secrets():
        if secret and len(secret) >= 4:
          patterns.add(secret)
    return patterns


def create_default_credential_proxy(
  env: Optional[Dict[str, str]] = None,
  rw_urls: Optional[str] = None
) -> CredentialProxy:
  """Inspects host environment variables and creates a CredentialProxy with standard routes."""
  e = os.environ if env is None else env
  raw_rw = rw_urls if rw_urls is not None else e.get("CHATTY_RW_URLS")
  rw_filter = RWUrlFilter(raw_rw)
  proxy = CredentialProxy(rw_filter=rw_filter)

  storage_rw_methods = frozenset({"GET", "HEAD", "PUT", "POST", "DELETE"})
  storage_ro_methods = frozenset({"GET", "HEAD"})
  has_rw = bool(rw_filter.patterns)
  allowed_storage = storage_rw_methods if has_rw else storage_ro_methods
  active_rw_filter = rw_filter if has_rw else None

  # Hugging Face
  hf_token = e.get("HF_TOKEN") or e.get("HUGGING_FACE_HUB_TOKEN")
  if hf_token:
    proxy.add_route(ProxyRoute(
      prefix="/hf",
      target_base_url="https://huggingface.co",
      auth_provider=BearerTokenProvider(hf_token),
      container_env_var="HF_ENDPOINT",
      allowed_methods=frozenset({"GET", "HEAD"})
    ))

  # OpenAI / Compatible LLM
  openai_key = e.get("OPENAI_API_KEY")
  if openai_key:
    proxy.add_route(ProxyRoute(
      prefix="/openai",
      target_base_url="https://api.openai.com",
      auth_provider=BearerTokenProvider(openai_key),
      container_env_var="OPENAI_BASE_URL",
      allowed_methods=frozenset({"GET", "POST"})
    ))

  # Anthropic
  anthropic_key = e.get("ANTHROPIC_API_KEY")
  if anthropic_key:
    proxy.add_route(ProxyRoute(
      prefix="/anthropic",
      target_base_url="https://api.anthropic.com",
      auth_provider=HeaderAuthProvider(
        headers=lambda k=anthropic_key: {"x-api-key": k, "anthropic-version": "2023-06-01"},
        secrets=lambda k=anthropic_key: [k]
      ),
      container_env_var="ANTHROPIC_BASE_URL",
      allowed_methods=frozenset({"GET", "POST"})
    ))

  # Cloudflare R2
  cf_account = e.get("CLOUDFLARE_ACCOUNT_ID")
  r2_ak = e.get("R2_ACCESS_KEY_ID")
  r2_sk = e.get("R2_SECRET_ACCESS_KEY")
  if cf_account and r2_ak and r2_sk:
    proxy.add_route(ProxyRoute(
      prefix="/r2",
      target_base_url=f"https://{cf_account}.r2.cloudflarestorage.com",
      auth_provider=SigV4AuthProvider(access_key=r2_ak, secret_key=r2_sk, service="s3", region="auto"),
      container_env_var="AWS_ENDPOINT_URL_S3",
      allowed_methods=allowed_storage,
      rw_filter=active_rw_filter
    ))

  # AWS S3 (when standard AWS credentials are set and distinct from R2)
  aws_ak = e.get("AWS_ACCESS_KEY_ID")
  aws_sk = e.get("AWS_SECRET_ACCESS_KEY")
  if aws_ak and aws_sk and not (cf_account and aws_ak == r2_ak):
    aws_region = e.get("AWS_DEFAULT_REGION", "us-east-1")
    s3_host = f"https://s3.{aws_region}.amazonaws.com" if aws_region != "us-east-1" else "https://s3.amazonaws.com"
    proxy.add_route(ProxyRoute(
      prefix="/s3",
      target_base_url=s3_host,
      auth_provider=SigV4AuthProvider(access_key=aws_ak, secret_key=aws_sk, service="s3", region=aws_region),
      container_env_var="AWS_ENDPOINT_URL",
      allowed_methods=allowed_storage,
      rw_filter=active_rw_filter
    ))

  # Google Cloud Storage
  gcp_creds = e.get("GOOGLE_APPLICATION_CREDENTIALS")
  if gcp_creds or e.get("ENABLE_GCS_PROXY") == "1":
    try:
      gcs_scopes = (
        ["https://www.googleapis.com/auth/devstorage.read_write"]
        if (has_rw and rw_filter.has_gcs_rules())
        else ["https://www.googleapis.com/auth/devstorage.read_only"]
      )
      proxy.add_route(ProxyRoute(
        prefix="/gcs",
        target_base_url="https://storage.googleapis.com",
        auth_provider=GCSAuthProvider(scopes=gcs_scopes),
        container_env_var="STORAGE_EMULATOR_HOST",
        allowed_methods=allowed_storage,
        rw_filter=active_rw_filter
      ))
    except Exception as ex:
      logger.debug(f"GCS proxy route skipped: {ex}")

  # GitHub
  gh_token = e.get("GITHUB_TOKEN") or e.get("GH_TOKEN")
  if gh_token:
    proxy.add_route(ProxyRoute(
      prefix="/github",
      target_base_url="https://api.github.com",
      auth_provider=BearerTokenProvider(gh_token),
      container_env_var="GITHUB_API_URL",
      allowed_methods=frozenset({"GET", "HEAD"})
    ))

  # Weights & Biases
  wandb_key = e.get("WANDB_API_KEY")
  if wandb_key:
    import base64
    basic_val = base64.b64encode(f"api:{wandb_key}".encode("utf-8")).decode("utf-8")
    proxy.add_route(ProxyRoute(
      prefix="/wandb",
      target_base_url="https://api.wandb.ai",
      auth_provider=HeaderAuthProvider(
        headers=lambda v=basic_val: {"Authorization": f"Basic {v}"},
        secrets=lambda k=wandb_key: [k]
      ),
      container_env_var="WANDB_BASE_URL",
      allowed_methods=frozenset({"GET", "POST", "HEAD"})
    ))

  return proxy
