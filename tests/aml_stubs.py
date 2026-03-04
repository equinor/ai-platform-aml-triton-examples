"""
Lightweight stubs for the AML inference server HTTP classes.

Injected into sys.modules by conftest.py so that scoring scripts can import
AMLRequest, AMLResponse, and rawhttp without the real azureml packages.
"""


class AMLResponse:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code


class AMLRequest:
    def __init__(self, method="POST", path="/score", body=b"", args=None, headers=None):
        self.method = method
        self.path = path
        _args = args or {}
        qs = "&".join(f"{k}={v}" for k, v in _args.items())
        self.full_path = f"{path}?{qs}" if qs else path
        self._body = body if isinstance(body, bytes) else body.encode()
        self.args = _args
        self.headers = headers or {}
        self.data = self._body

    def get_data(self, as_text=False, **_):
        return self._body.decode() if as_text else self._body


def rawhttp(fn):
    """Identity decorator — returns the function unchanged."""
    return fn
