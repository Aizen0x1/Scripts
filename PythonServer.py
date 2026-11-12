#!/usr/bin/env python3
"""
secure_server.py — hardened static file server for the Aizen blog.

Security properties
-------------------
* Threaded, but with a bounded worker pool so a connection flood cannot
  spawn unbounded threads.
* Per-connection socket timeout (Slowloris resistant) and a larger listen
  backlog.
* Strict HTTP methods: GET, HEAD and OPTIONS only.
* No directory listings, no dotfiles, no source/backup/config extensions.
* Symlink-safe: every served path is realpath-resolved and must stay inside
  the document root.
* Rejects NUL/control characters, backslashes and over-long paths with 4xx
  instead of raising an unhandled exception.
* Full security header set, including a strict self-only CSP (all JS, CSS,
  fonts and images are served locally).
* Server banner omitted; error pages do not reflect user input.
* Optional TLS (TLS 1.2+) with HSTS enabled automatically when TLS is active.

Usage
-----
    python3 secure_server.py                 # serves this directory on :8000
    python3 secure_server.py 8000            # positional port (old style)
    python3 secure_server.py --port 8010
    python3 secure_server.py --cert cert.pem --key key.pem
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import socketserver
import ssl
import sys
import threading
import urllib.parse
from datetime import timezone
from email.utils import parsedate_to_datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_PATH_LENGTH = 2048
DEFAULT_MAX_WORKERS = 64
DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"
SOCKET_TIMEOUT = 15          # seconds; kills idle/slow clients
REQUEST_QUEUE_SIZE = 128     # listen backlog

# Only these extensions may be served. Anything else (including .py, .sh,
# .env, .git, .bak, .sql, .pem, .key, ...) is treated as 404.
ALLOWED_EXTENSIONS = frozenset({
    ".html", ".htm", ".css", ".js", ".md", ".txt", ".json", ".webmanifest",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".otf",
})

ASSET_EXTENSIONS = frozenset({
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".otf",
})

CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "manifest-src 'self'; "
    "media-src 'none'; "
    "worker-src 'none'; "
    "frame-src 'none'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)

PERMISSIONS_POLICY = (
    "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
    "magnetometer=(), microphone=(), payment=(), usb=(), interest-cohort=()"
)


# ---------------------------------------------------------------------------
# Bounded threading server
# ---------------------------------------------------------------------------

class BoundedThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """ThreadingHTTPServer with a bounded number of concurrent workers."""

    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True
    request_queue_size = REQUEST_QUEUE_SIZE

    def __init__(self, server_address, request_handler_class,
                 bind_and_activate=True, max_workers=DEFAULT_MAX_WORKERS):
        self._worker_slots = threading.BoundedSemaphore(max_workers)
        self.tls_enabled = False
        super().__init__(server_address, request_handler_class, bind_and_activate)

    def process_request(self, request, client_address):
        if not self._worker_slots.acquire(blocking=False):
            # Load shedding: never queue an unbounded number of handler threads.
            with contextlib.suppress(OSError):
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\n"
                    b"Content-Length: 0\r\n\r\n"
                )
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class SecureHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = SOCKET_TIMEOUT
    server_version = "securehost"
    sys_version = ""

    # -- lifecycle ----------------------------------------------------------

    def __init__(self, *args, directory=None, **kwargs):
        # Resolve the docroot *before* SimpleHTTPRequestHandler starts handling
        # the request in its own __init__.
        self.directory_real = os.path.realpath(
            directory if directory is not None else os.getcwd()
        )
        super().__init__(*args, directory=directory, **kwargs)

    # -- response metadata --------------------------------------------------

    def version_string(self):
        return "securehost"

    def send_response(self, code, message=None):
        # Do not emit a "Server:" banner at all.
        self.log_request(code)
        self.send_response_only(code, message)
        self.send_header("Date", self.date_time_string())

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", PERMISSIONS_POLICY)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Permitted-Cross-Domain-Policies", "none")
        self.send_header("Content-Security-Policy", CSP)

        cache = self._cache_control()
        if cache:
            self.send_header("Cache-Control", cache)

        if getattr(self.server, "tls_enabled", False):
            self.send_header(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )

        super().end_headers()

    def _cache_control(self):
        path = urllib.parse.urlsplit(self.path or "").path.lower()
        if path.endswith(tuple(ASSET_EXTENSIONS)):
            # Vendor JS/CSS, fonts and images are content-addressed by us and
            # can be cached; HTML/Markdown are always revalidated.
            return "public, max-age=86400"
        return "no-cache"

    # -- logging ------------------------------------------------------------

    def log_message(self, fmt, *args):
        # Strip control characters to prevent log/ANSI injection.
        try:
            message = fmt % args
        except Exception:
            message = str(fmt)
        message = "".join(
            ch if 32 <= ord(ch) < 127 else "?" for ch in str(message)
        )
        sys.stderr.write(
            "%s - - [%s] %s\n" % (
                self.address_string(),
                self.log_date_time_string(),
                message,
            )
        )

    # -- method policy ------------------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", "GET, HEAD, OPTIONS")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _method_not_allowed(self):
        body = b"405 Method Not Allowed\n"
        self.send_response(405)
        self.send_header("Allow", "GET, HEAD, OPTIONS")
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                self.wfile.write(body)
        self.close_connection = True

    def _has_request_body(self):
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding:
            return True
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) > 1:
            return True
        if not lengths:
            return False
        try:
            return int(lengths[0]) > 0
        except (TypeError, ValueError):
            return True

    def _serve_safely(self, handler):
        # A client that disconnects mid-response is normal; do not let it
        # produce a traceback or hold a worker slot.
        try:
            handler()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True

    def do_GET(self):
        # Bodies on GET/HEAD are not expected and can desynchronise a
        # keep-alive connection; reject them and close.
        if self._has_request_body():
            self.send_error(400, "Unexpected request body")
            return
        self._serve_safely(super().do_GET)

    def do_HEAD(self):
        if self._has_request_body():
            self.send_error(400, "Unexpected request body")
            return
        self._serve_safely(super().do_HEAD)

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_TRACE = _method_not_allowed
    do_CONNECT = _method_not_allowed

    # -- request validation -------------------------------------------------

    def parse_request(self):
        ok = super().parse_request()
        if ok and self.request_version == "HTTP/0.9":
            # HTTP/0.9 has no headers and is not needed by this site.
            self.send_error(505)
            return False
        return ok

    def _path_error(self):
        raw = self.path or ""

        if len(raw) > MAX_PATH_LENGTH:
            return 414
        if not raw.startswith("/"):
            return 400
        if "\\" in raw:
            return 400

        for ch in raw:
            if ord(ch) < 0x20 or ord(ch) == 0x7F:
                return 400

        try:
            decoded = urllib.parse.unquote(raw, errors="strict")
        except (UnicodeDecodeError, ValueError):
            return 400

        if "\\" in decoded or "\x00" in decoded:
            return 400
        for ch in decoded:
            if ord(ch) < 0x20 or ord(ch) == 0x7F:
                return 400

        return None

    # -- file serving -------------------------------------------------------

    def list_directory(self, path):
        # Directory listings are disabled entirely.
        self.send_error(404)
        return None

    def send_head(self):
        error = self._path_error()
        if error is not None:
            self.send_error(error)
            return None

        path = self.translate_path(self.path)

        try:
            real = os.path.realpath(path)
        except (OSError, ValueError):
            self.send_error(400)
            return None

        root = self.directory_real
        try:
            if os.path.commonpath([root, real]) != root:
                self.send_error(404)
                return None
        except ValueError:
            self.send_error(404)
            return None

        # Block any dot-segment (.git, .env, .gitkeep, ...), even if the
        # filesystem has such a file.
        url_path = urllib.parse.urlsplit(self.path).path
        segments = [segment for segment in url_path.split("/") if segment]
        if any(segment.startswith(".") for segment in segments):
            self.send_error(404)
            return None

        # Directory handling: redirect to a trailing slash, then only serve
        # an index.html if present. Never generate a listing.
        if os.path.isdir(real):
            if not url_path.endswith(("/", "%2f", "%2F")):
                parts = urllib.parse.urlsplit(self.path)
                new_url = urllib.parse.urlunsplit((
                    parts.scheme,
                    parts.netloc,
                    parts.path + "/",
                    parts.query,
                    parts.fragment,
                ))
                self.send_response(301)
                self.send_header("Location", new_url)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None

            index = os.path.join(real, "index.html")
            if not os.path.isfile(index):
                self.send_error(404)
                return None
            real = os.path.realpath(index)
            try:
                if os.path.commonpath([root, real]) != root:
                    self.send_error(404)
                    return None
            except ValueError:
                self.send_error(404)
                return None

        if not os.path.isfile(real):
            self.send_error(404)
            return None

        extension = os.path.splitext(real)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            self.send_error(404)
            return None

        try:
            file_obj = open(real, "rb")
        except (OSError, ValueError):
            self.send_error(404)
            return None

        try:
            stat = os.fstat(file_obj.fileno())

            # Conditional GET (same semantics as the stdlib handler).
            ims = self.headers.get("If-Modified-Since")
            if ims:
                try:
                    dt = parsedate_to_datetime(ims)
                    if dt is not None:
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if int(stat.st_mtime) <= int(dt.timestamp()):
                            file_obj.close()
                            self.send_response(304)
                            self.send_header(
                                "Last-Modified",
                                self.date_time_string(stat.st_mtime),
                            )
                            self.end_headers()
                            return None
                except (TypeError, ValueError, OverflowError):
                    pass

            content_type = self.guess_type(real) or "application/octet-stream"
            if (content_type.startswith("text/")
                    or content_type in (
                        "application/javascript",
                        "application/json",
                        "application/manifest+json",
                        "image/svg+xml",
                    )):
                content_type += "; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return file_obj
        except Exception:
            file_obj.close()
            raise

    # -- generic, non-reflective error pages --------------------------------

    def send_error(self, code, message=None, explain=None):
        try:
            short = self.responses[code][0]
        except KeyError:
            short = "Error"

        title = f"{code} {short}"
        body = (
            "<!DOCTYPE html><html lang=\"en\"><head>"
            "<meta charset=\"utf-8\"><title>"
            f"{title}"
            "</title></head><body><h1>"
            f"{title}"
            "</h1><p>The requested resource is not available.</p>"
            "</body></html>"
        ).encode("utf-8")

        self.send_response(code, short)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()

        if (self.command != "HEAD"
                and code >= 200
                and code not in (204, 304)):
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                self.wfile.write(body)

        self.close_connection = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Hardened static server for the Aizen blog",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help=f"Port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "port_pos", nargs="?", type=int, metavar="PORT",
        help="Port as a positional argument (compatibility with the old server)",
    )
    parser.add_argument(
        "--root", default=str(default_root),
        help="Document root (default: the directory containing this file)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Maximum concurrent request threads (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument("--cert", help="TLS certificate (PEM) to enable HTTPS")
    parser.add_argument("--key", help="TLS private key (PEM) to enable HTTPS")
    return parser


def main():
    args = build_parser().parse_args()

    if args.port is not None and args.port_pos is not None:
        sys.exit("error: give the port either positionally or with --port, not both")
    port = args.port if args.port is not None else args.port_pos
    if port is None:
        port = DEFAULT_PORT

    if not (0 < port < 65536):
        sys.exit("error: port must be between 1 and 65535")
    if args.max_workers < 1:
        sys.exit("error: --max-workers must be >= 1")
    if bool(args.cert) != bool(args.key):
        sys.exit("error: --cert and --key must be supplied together")

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"error: document root does not exist: {root}")

    handler = functools.partial(SecureHandler, directory=str(root))

    httpd = BoundedThreadingHTTPServer(
        (args.host, port),
        handler,
        max_workers=args.max_workers,
    )

    tls = bool(args.cert and args.key)
    if tls:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.cert, args.key)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        httpd.tls_enabled = True

    scheme = "https" if tls else "http"
    print(f"securehost: serving {root}")
    print(f"securehost: {scheme}://{args.host}:{port}/")
    print(f"securehost: workers={args.max_workers} timeout={SOCKET_TIMEOUT}s")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("securehost: WARNING: bound to a non-loopback address.")
        print("securehost:          Use TLS + a reverse proxy for remote access.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nsecurehost: shutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
