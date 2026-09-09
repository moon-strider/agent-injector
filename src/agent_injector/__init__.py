"""Agent Injector's command line entry point."""

__version__ = "0.2.0"


def main() -> None:
    import argparse
    import asyncio
    import json
    import logging
    import sys

    from .config import ConfigurationError, load_settings
    from .models import Input
    from .server import Application

    parser = argparse.ArgumentParser(
        description="Run Claude Code tasks through a local stdio MCP server"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--check", action="store_true", help="print configuration diagnostics without inference"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )
    logging.getLogger("agent_injector").setLevel(logging.INFO)
    try:
        app = Application(load_settings())
        if args.check:
            print(json.dumps(asyncio.run(app.dispatch("llm_status", Input())), indent=2))
        else:
            asyncio.run(app.run())
    except ConfigurationError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    except KeyboardInterrupt:
        pass
