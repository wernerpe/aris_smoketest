#!/usr/bin/env python3
"""Send a command to the running Blender instance's MCP-style socket on 127.0.0.1:9876."""
import json
import socket
import sys


def send_command(cmd_type, params=None, timeout=60.0):
    payload = json.dumps({"type": cmd_type, "params": params or {}}).encode("utf-8")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(("127.0.0.1", 9876))
    s.sendall(payload)

    chunks = []
    while True:
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
        # Try to parse what we have so far; if it's complete JSON we're done.
        try:
            data = json.loads(b"".join(chunks).decode("utf-8"))
            s.close()
            return data
        except json.JSONDecodeError:
            continue
    s.close()
    raw = b"".join(chunks).decode("utf-8", errors="replace")
    raise RuntimeError(f"Incomplete/unparseable response: {raw[:2000]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "get_scene_info"
    params = {}
    if len(sys.argv) > 2:
        if sys.argv[2] == "--file":
            with open(sys.argv[3]) as f:
                params = {"code": f.read()}
        else:
            params = json.loads(sys.argv[2])
    result = send_command(cmd, params)
    print(json.dumps(result, indent=2, default=str)[:8000])
