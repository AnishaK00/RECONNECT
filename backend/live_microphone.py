"""Live microphone streaming entrypoint for RECONNECT.

Directly invokes the live microphone streaming client.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from live_microphone_client import main

if __name__ == "__main__":
    main()
