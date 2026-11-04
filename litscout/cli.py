"""Command line entry point: `litscout serve` or `litscout seed-demo`."""

from __future__ import annotations

import argparse

from .config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="litscout", description="Literature search, screening and review drafting")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="Run the web app")
    s.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow other computers (set a password first)")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("seed-demo", help="Create a sample project from the offline demo corpus")
    args = parser.parse_args()

    if args.cmd == "serve":
        import uvicorn

        settings = get_settings()
        if args.host not in ("127.0.0.1", "localhost") and not settings.access_password:
            print("Warning: serving on the network without LITSCOUT_ACCESS_PASSWORD. Anyone on the network can use this app.")
        uvicorn.run("litscout.main:app", host=args.host, port=args.port)
    elif args.cmd == "seed-demo":
        from .db import Store
        from .demo_project import seed_demo_project

        pid = seed_demo_project(Store(get_settings().db_path))
        print(f"Created demo project #{pid}. Run `litscout serve` and open http://127.0.0.1:8000/projects/{pid}")


if __name__ == "__main__":
    main()
