"""Print stored audio events. Run with: python3 -m app.events.read_events."""

import argparse
import json

from app.events.db import DB_PATH, init_db, list_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stored audio events as JSON")
    parser.add_argument("--status", choices=("open", "closed"), help="filter events")
    args = parser.parse_args()
    init_db()
    print(f"Database: {DB_PATH}")
    print(json.dumps(list_events(status=args.status), indent=2))


if __name__ == "__main__":
    main()
