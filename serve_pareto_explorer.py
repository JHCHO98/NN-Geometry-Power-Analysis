"""Serve the lightweight Pareto explorer and its candidate data locally."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from select_certified_pareto import mean_pareto_mask


class ExplorerHandler(SimpleHTTPRequestHandler):
    """Serve static explorer files and build a compact JSON response on demand."""

    search_dir: Path
    predictions_path: Path
    cached_mtime_ns: int | None = None
    cached_data: bytes | None = None

    def do_GET(self) -> None:  # noqa: N802 - required base-class method name
        if urlparse(self.path).path == "/api/data":
            self.serve_data()
            return
        if urlparse(self.path).path == "/":
            self.path = "/pareto_structures.html"
        super().do_GET()

    def translate_path(self, path: str) -> str:
        requested = Path(urlparse(path).path.lstrip("/")).as_posix()
        target = (self.search_dir / requested).resolve()
        if self.search_dir not in target.parents and target != self.search_dir:
            return str(self.search_dir / "__forbidden__")
        return str(target)

    @classmethod
    def build_data(cls) -> bytes:
        modified = cls.predictions_path.stat().st_mtime_ns
        if cls.cached_data is not None and cls.cached_mtime_ns == modified:
            return cls.cached_data

        frame = pd.read_csv(cls.predictions_path, encoding="utf-8-sig")
        accuracy_column = "ensemble_accuracy_mean_percent"
        energy_column = "ensemble_energy_mean_j"
        latency_column = "ensemble_latency_mean_ms"
        required = {
            "candidate_id", "pattern", "depth", "pool_count", "parameter_count", "channels", "pools",
            accuracy_column, energy_column, latency_column,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Prediction CSV is missing: {sorted(missing)}")

        accuracy = pd.to_numeric(frame[accuracy_column], errors="raise").to_numpy(float)
        energy_mj = pd.to_numeric(frame[energy_column], errors="raise").to_numpy(float) * 1000.0
        global_pareto = mean_pareto_mask(accuracy, energy_mj)
        pattern_pareto = np.zeros(len(frame), dtype=bool)
        patterns = frame["pattern"].astype(str).to_numpy()
        for pattern in np.unique(patterns):
            positions = np.flatnonzero(patterns == pattern)
            pattern_pareto[positions] = mean_pareto_mask(accuracy[positions], energy_mj[positions])

        records = []
        for index, row in frame.iterrows():
            depth = int(row["depth"])
            pool_count = int(row["pool_count"])
            records.append(
                {
                    "id": str(row["candidate_id"]), "pattern": str(row["pattern"]),
                    "depth": depth, "poolCount": pool_count, "poolRatio": pool_count / depth,
                    "parameters": int(row["parameter_count"]), "channels": str(row["channels"]),
                    "pools": str(row["pools"]), "accuracy": float(accuracy[index]),
                    "energy": float(energy_mj[index]),
                    "latency": float(row[latency_column]), "globalPareto": bool(global_pareto[index]),
                    "patternPareto": bool(pattern_pareto[index]),
                }
            )
        cls.cached_data = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        cls.cached_mtime_ns = modified
        return cls.cached_data

    def serve_data(self) -> None:
        try:
            data = self.build_data()
        except (OSError, ValueError) as error:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--search-dir", type=Path, default=Path("measurements/search"))
    parser.add_argument("--predictions", type=Path, default=Path("measurements/search/candidate_predictions.csv"))
    args = parser.parse_args()
    if not args.search_dir.is_dir() or not args.predictions.is_file():
        raise FileNotFoundError("Expected measurements/search/ and candidate_predictions.csv.")
    ExplorerHandler.search_dir = args.search_dir.resolve()
    ExplorerHandler.predictions_path = args.predictions.resolve()
    server = ThreadingHTTPServer((args.host, args.port), ExplorerHandler)
    print(f"Open http://{args.host}:{args.port}/ in a browser. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
