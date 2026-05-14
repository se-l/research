import argparse
import json
import pickle
from pathlib import Path

from derivatives.surfaces.processors import get_v_ivs

def load_payload(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_result(path: str, result):
    with open(path, "wb") as f:
        pickle.dump(result, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="Path to JSON payload")
    parser.add_argument("--out", required=True, help="Path to write result pickle")
    args = parser.parse_args()

    payload = load_payload(args.payload)

    # Reconstruct only what you need on the remote node.
    # Example payload fields:
    # {
    #   "equity_symbol": "DAL",
    #   "dates": ["2024-01-01"],
    #   "resolution": "second",
    #   "seq_ret_threshold": 0.002,
    #   "arb_free": false,
    #   "seq_ret_threshold_surface": null
    # }

    v_ivs = get_v_ivs(
        payload["equity_symbol"],
        payload["dates"],
        payload["resolution"],
        payload["seq_ret_threshold"],
        payload["arb_free"],
        payload["seq_ret_threshold_surface"],
    )

    # If calibration is part of the remote task:
    # calibrate_yield_curve_and_store(v_ivs, calc_date, equity, seq_ret_threshold=...)

    save_result(args.out, v_ivs)
    print(f"Saved result to {args.out}")

if __name__ == "__main__":
    main()