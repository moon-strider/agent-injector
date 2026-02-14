__version__ = "0.1.0"

from . import server
import asyncio


def main():
    asyncio.run(server.main_async())


__all__ = ["main", "server"]
