#!/usr/bin/env python3
"""Sample local host for the OpenCode Config Editor's local mode.

Serves ../index.html (or --page) with window.__OPENCODE_LOCAL__ injected, so the editor edits
an OpenCode config file on this machine instead of importing and downloading files. When the
page saves, the result comes back here: the terminal shows a diff and asks before writing, and
a backup of the old file is kept. Standard library only, Python 3.8+.

Edit the global config:

    python3 examples/local_host.py

Add (or select) a provider for an OpenAI-compatible gateway, the way a setup script would.
The key is read from the environment, is used for the gateway's /models list and is filled in
for editor requests sent to the gateway; the config only gets an {env:} reference to it:

    export MY_GATEWAY_KEY=sk-...
    python3 examples/local_host.py --provider my-gateway --name "My Gateway" \\
        --base-url https://gateway.example.com/v1 --key-env MY_GATEWAY_KEY

Use --dry-run to show the diff without writing, and --no-browser to only print the URL.

Host contract (also described next to `const LOCAL` in index.html):

    GET  {api}/config          -> {path, text, planned?, models?}
    ANY  {api}/proxy?url=URL   -> forwards the editor's /models and test requests
    POST {api}/save   {text}   -> the edited file text
    POST {api}/cancel          -> the user cancelled

The server listens on 127.0.0.1 and every route is under a random token, so other local
pages and programs cannot use it, including its proxy.
"""

import argparse
import datetime
import difflib
import http.server
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

SCHEMA = "https://opencode.ai/config.json"
HOOK = "window.__OPENCODE_LOCAL__"
MAX_BODY = 1 << 20
MAX_RESPONSE = 16 << 20
USER_AGENT = "opencode-config-editor-local-host/1.0"
# Headers the browser adds that should not be forwarded upstream.
DROP_HEADERS = {
    "host", "origin", "referer", "cookie", "connection", "content-length", "accept-encoding",
    "accept-language", "user-agent", "keep-alive", "transfer-encoding", "te", "upgrade",
    "cache-control", "pragma", "priority", "dnt",
}


class HostError(Exception):
    pass


# ---------------------------------------------------------------- config file
def parse_jsonc(text):
    """Parse OpenCode's JSONC: JSON plus // and /* */ comments and trailing commas."""
    text = text.lstrip("﻿")
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise HostError("unterminated /* comment")
            out.append(" ")
            i = j + 2
        else:
            if c in "}]":
                k = len(out) - 1
                while k >= 0 and out[k].isspace():
                    k -= 1
                if k >= 0 and out[k] == ",":
                    del out[k]
            out.append(c)
            i += 1
    stripped = "".join(out)
    if not stripped.strip():
        return {}
    try:
        return json.loads(stripped)
    except ValueError as exc:
        raise HostError("not valid JSON: %s" % exc)


def default_config_path():
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode"
    for name in ("opencode.jsonc", "opencode.json"):
        if (base / name).exists():
            return base / name
    return base / "opencode.json"


def provider_entry(config, pid):
    """The provider's entry and its format: V2 "providers" wins over V1 "provider"."""
    for key, fmt in (("providers", "v2"), ("provider", "v1")):
        entries = config.get(key)
        if isinstance(entries, dict) and isinstance(entries.get(pid), dict):
            return entries[pid], fmt
    return None, None


def provider_base_url(config, pid):
    entry, fmt = provider_entry(config, pid)
    if entry is None:
        return None
    options = entry.get("settings" if fmt == "v2" else "options")
    url = options.get("baseURL") if isinstance(options, dict) else None
    return url if isinstance(url, str) and url.startswith(("http://", "https://")) else None


def plan_config(config, args):
    """The config the editor starts from: the file, plus the --provider entry if it is new."""
    planned = json.loads(json.dumps(config))
    planned.setdefault("$schema", SCHEMA)
    if args.provider and args.base_url and provider_entry(planned, args.provider)[0] is None:
        options = {"baseURL": args.base_url}
        if args.key_env:
            options["apiKey"] = "{env:%s}" % args.key_env
        entry = {"npm": "@ai-sdk/openai-compatible", "name": args.name or args.provider,
                 "options": options, "models": {}}
        if not isinstance(planned.get("provider"), dict):
            planned["provider"] = {}
        planned["provider"][args.provider] = entry
    return planned


def detect_target():
    """'v1' or 'v2' from the installed `opencode --version`, or None."""
    try:
        res = subprocess.run(["opencode", "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    major = res.stdout.strip().lstrip("v").split(".")[0]
    return {"1": "v1", "2": "v2"}.get(major)


def write_config(path, text):
    """Back up the old file, then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    mode = 0o644
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_name("%s.bak-%s" % (path.name, stamp))
        shutil.copy2(path, backup)
        mode = path.stat().st_mode & 0o777
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return backup


# ---------------------------------------------------------------- upstream requests
class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: they would carry the added credentials to another URL."""

    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def under(url, base):
    base = base.rstrip("/")
    return url == base or url.startswith(base + "/") or url.startswith(base + "?")


def fetch_models(base_url, key):
    """Model IDs from an OpenAI-compatible {base}/models, or [] with a warning."""
    req = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"User-Agent": USER_AGENT})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with OPENER.open(req, timeout=30) as res:
            data = json.loads(res.read(MAX_RESPONSE))
        return sorted({m["id"] for m in data.get("data", []) if isinstance(m, dict) and isinstance(m.get("id"), str)})
    except (urllib.error.URLError, OSError, ValueError, AttributeError) as exc:
        print("warning: could not list models from %s/models: %s" % (base_url.rstrip("/"), exc), file=sys.stderr)
        return []


# ---------------------------------------------------------------- server
class State:
    def __init__(self, page, prefix, config_doc, base_url, key, port_holder):
        self.page = page
        self.prefix = prefix
        self.config_doc = config_doc
        self.base_url = base_url
        self.key = key
        self.port_holder = port_holder
        self.result = None
        self.done = threading.Event()


def make_handler(state, loopback):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "local-host"

        def log_message(self, fmt, *args):  # keep the terminal for the user
            pass

        def send(self, status, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def error(self, status, message):
            self.send(status, {"error": message})

        def read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise HostError("request body too large")
            return self.rfile.read(length) if length else b""

        def handle_any(self, method):
            port = state.port_holder[0]
            host = self.headers.get("Host", "")
            # DNS rebinding: a loopback server only answers to loopback host names.
            if loopback and host not in ("127.0.0.1:%d" % port, "localhost:%d" % port):
                return self.error(403, "unexpected Host header")
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + host:
                return self.error(403, "cross-origin request")
            url = urllib.parse.urlsplit(self.path)
            if not url.path.startswith(state.prefix):
                return self.error(404, "not found")
            route = url.path[len(state.prefix):]
            try:
                if method == "GET" and route in ("", "index.html"):
                    return self.send(200, state.page, "text/html; charset=utf-8")
                if method == "GET" and route == "api/config":
                    return self.send(200, state.config_doc)
                if route == "api/proxy":
                    target = urllib.parse.parse_qs(url.query).get("url", [""])[0]
                    return self.proxy(method, target, self.read_body())
                if method == "POST" and route in ("api/save", "api/cancel"):
                    body = self.read_body()
                    text = None
                    if route == "api/save":
                        try:
                            text = json.loads(body or b"{}").get("text")
                        except (ValueError, AttributeError):
                            text = None
                        if not isinstance(text, str):
                            return self.error(400, "missing text")
                        if not isinstance(parse_jsonc(text), dict):
                            return self.error(400, "the edited config is not a JSON object")
                    self.send(200, {"ok": True})
                    state.result = text
                    state.done.set()
                    return None
                return self.error(404, "not found")
            except HostError as exc:
                return self.error(400, str(exc))

        def proxy(self, method, target, data):
            parts = urllib.parse.urlsplit(target)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                return self.error(400, "proxy url must be http(s)")
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in DROP_HEADERS and not k.lower().startswith(("sec-", "proxy-"))}
            headers["User-Agent"] = USER_AGENT
            lower = {k.lower() for k in headers}
            # Fill in the key for the host's own provider when the page has none (its config
            # only holds an {env:} reference, which the browser cannot resolve).
            if state.key and state.base_url and under(target, state.base_url) \
                    and not lower & {"authorization", "x-api-key"}:
                if "anthropic-version" in lower:
                    headers["x-api-key"] = state.key
                else:
                    headers["Authorization"] = "Bearer " + state.key
            req = urllib.request.Request(target, data=data or None, method=method, headers=headers)
            try:
                with OPENER.open(req, timeout=60) as res:
                    status, ctype, body = res.status, res.headers.get("Content-Type"), res.read(MAX_RESPONSE + 1)
            except urllib.error.HTTPError as exc:
                status, ctype, body = exc.code, exc.headers.get("Content-Type"), exc.read(MAX_RESPONSE + 1)
            except (urllib.error.URLError, OSError) as exc:
                return self.error(502, "cannot reach %s: %s" % (target, getattr(exc, "reason", exc)))
            if len(body) > MAX_RESPONSE:
                return self.error(502, "upstream response too large")
            return self.send(status, body, ctype or "application/octet-stream")

        def do_GET(self):
            self.handle_any("GET")

        def do_POST(self):
            self.handle_any("POST")

        def do_PUT(self):
            self.handle_any("PUT")

        def do_DELETE(self):
            self.handle_any("DELETE")

    return Handler


def load_page(source):
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(urllib.request.Request(source, headers={"User-Agent": USER_AGENT}), timeout=30) as res:
            page = res.read().decode("utf-8")
    else:
        page = Path(source).read_text(encoding="utf-8")
    if HOOK not in page:
        raise HostError("%s does not support local mode (no %s)" % (source, HOOK))
    return page


def inject(page, local):
    script = "<script>%s = %s;</script>\n" % (HOOK, json.dumps(local).replace("</", "<\\/"))
    if "</head>" not in page:
        raise HostError("the editor page has no </head>")
    return page.replace("</head>", script + "</head>", 1).encode("utf-8")


def summarize(old_text, new_text, path, pid):
    diff = difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                str(path) + " (current)", str(path) + " (edited)")
    sys.stdout.writelines(line if line.endswith("\n") else line + "\n" for line in diff)
    new = parse_jsonc(new_text)
    disabled = new.get("disabled_providers") if isinstance(new.get("disabled_providers"), list) else []
    if pid and provider_entry(new, pid)[0] is None:
        print("warning: provider %r is no longer in the config" % pid)
    if pid and pid in disabled:
        print("warning: provider %r is in disabled_providers" % pid)
    for key in ("model", "small_model"):
        value = new.get(key)
        if isinstance(value, str) and value.split("/", 1)[0] in disabled:
            print("warning: %s %r points at a disabled provider" % (key, value))


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Edit an OpenCode config with the editor's local mode.")
    ap.add_argument("config", nargs="?", type=Path, help="config file (default: the global OpenCode config)")
    ap.add_argument("--page", default=str(here.parent / "index.html"),
                    help="editor page, a file or an http(s) URL (default: %(default)s)")
    ap.add_argument("--provider", help="provider ID to select (and add, with --base-url)")
    ap.add_argument("--name", help="display name for a provider added with --base-url")
    ap.add_argument("--base-url", help="OpenAI-compatible base URL for --provider, e.g. https://gw.example/v1")
    ap.add_argument("--key-env", help="environment variable with the --provider API key")
    ap.add_argument("--target", choices=("v1", "v2"), help="OpenCode version (default: from `opencode --version`)")
    ap.add_argument("--port", type=int, default=0, help="port (default: any free port)")
    ap.add_argument("--no-browser", action="store_true", help="only print the editor URL")
    ap.add_argument("--dry-run", action="store_true", help="show the changes without writing")
    ap.add_argument("-y", "--yes", action="store_true", help="write without asking")
    args = ap.parse_args(argv)

    try:
        if args.base_url and not args.provider:
            raise HostError("--base-url needs --provider")
        key = None
        if args.key_env:
            key = os.environ.get(args.key_env)
            if not key:
                raise HostError("environment variable %s is not set" % args.key_env)
        path = (args.config or default_config_path()).expanduser().resolve()
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        try:
            config = parse_jsonc(original)
        except HostError as exc:
            raise HostError("%s: %s" % (path, exc))
        if not isinstance(config, dict):
            raise HostError("%s does not contain a JSON object" % path)
        planned = plan_config(config, args)
        base_url = provider_base_url(planned, args.provider) if args.provider else None
        if args.provider and provider_entry(planned, args.provider)[0] is None:
            print("warning: provider %r is not in the config; pass --base-url to add it" % args.provider, file=sys.stderr)
        page_source = load_page(args.page)
    except (HostError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    prefix = "/%s/" % secrets.token_urlsafe(24)
    local = {"api": prefix + "api", "path": str(path)}
    if args.provider:
        local["providerId"] = args.provider
    target = args.target or detect_target()
    if target:
        local["target"] = target
    config_doc = {"path": str(path), "text": original, "planned": planned}
    if base_url:
        config_doc["models"] = fetch_models(base_url, key)

    port_holder = [0]
    state = State(inject(page_source, local), prefix, config_doc, base_url, key, port_holder)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state, loopback=True))
    port_holder[0] = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d%s" % (port_holder[0], prefix)
    print("Editing %s\nOpen %s\n(press \"Save to system\" or Cancel in the page; Ctrl+C here cancels)" % (path, url),
          flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while not state.done.wait(0.5):
            pass
    except KeyboardInterrupt:
        print()
        state.result = None
    finally:
        server.shutdown()
        server.server_close()

    text = state.result
    if text is None:
        print("Cancelled; nothing written.")
        return 1
    if text == original:
        print("No changes.")
        return 0
    summarize(original, text, path, args.provider)
    if args.dry_run:
        print("Dry run; nothing written.")
        return 0
    if not args.yes:
        try:
            answer = input("Write %s? [y/N] " % path)
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing written.")
            return 1
    backup = write_config(path, text)
    print("Wrote %s%s" % (path, " (backup: %s)" % backup if backup else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
