from __future__ import annotations

import argparse
import logging
import pathlib
import socket
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI_ROOT = (ROOT / "../RealWorld-Pi").resolve()
for path in (
    OPENPI_ROOT / "src",
    OPENPI_ROOT / "packages/openpi-client/src",
    OPENPI_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.policies import policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as openpi_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", default="/data1/vla-data/openpi/openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    train_config = openpi_config.get_config(args.config)
    policy = policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    hostname = socket.gethostname()
    logging.info("Serving %s from %s on %s:%s", args.config, args.checkpoint_dir, hostname, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
