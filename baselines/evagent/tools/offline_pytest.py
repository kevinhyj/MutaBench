"""Disable Internet connections before pytest collection or optional imports."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import socket
import sys

# The selected environment must import this maintained checkout, even under -I.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Unrelated installed plugins may import providers or start tracing at collection.
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'


def no_internet(*_args, **_kwargs):
    raise RuntimeError("Internet access disabled by EVAgent offline test runner")


def local_only_connect(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        return no_internet()
    return original_connect(sock, address)


original_connect = socket.socket.connect
socket.socket.connect = local_only_connect
socket.socket.connect_ex = no_internet
socket.create_connection = no_internet
socket.getaddrinfo = no_internet

import pytest  # noqa: E402

plugins = []
options = []
if importlib.util.find_spec('pytest_socket') is not None:
    plugins += ['-p', 'pytest_socket']
    options += ['--disable-socket', '--allow-unix-socket']
if importlib.util.find_spec('pytest_asyncio') is not None:
    plugins += ['-p', 'pytest_asyncio.plugin']
arguments = sys.argv[1:]
paths = [] if any(arg.startswith('tests/') or arg == 'tests' for arg in arguments) else ['tests']
raise SystemExit(pytest.main([
    *paths, '-o', 'addopts=', '-m', 'not live', *plugins, *options, *arguments,
]))
