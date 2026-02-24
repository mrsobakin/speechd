from __future__ import annotations

import argparse
import logging
from importlib import resources

from speechd.config import Config
from speechd.daemon import SpeechDaemon


def get_service_unit() -> str:
    return resources.files("speechd").joinpath("speechd.service").read_text()


def install_service():
    from pathlib import Path

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "speechd.service"

    service_dir.mkdir(parents=True, exist_ok=True)
    service_file.write_text(get_service_unit())

    print(f"Installed: {service_file}")
    print("\nNext steps:")
    print("  1. Create ~/.config/speechd/config.toml with your API key")
    print("  2. Run: systemctl --user enable --now speechd")
    print("  3. Toggle recording: speechd-toggle")


def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text daemon")
    parser.add_argument(
        "--install-service", action="store_true", help="Install systemd user service"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.install_service:
        install_service()
        return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = Config.load()
    except RuntimeError as e:
        logging.error(str(e))
        raise SystemExit(1)

    daemon = SpeechDaemon(config)
    daemon.run()


if __name__ == "__main__":
    main()
