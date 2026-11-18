"""Write the HMM decoder configuration and artifact manifest."""
import json
from datetime import datetime, timezone
from pathlib import Path
from src.hmm_viterbi import HMMConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = HMMConfig()
    payload = {name: getattr(config, name) for name in config.__dataclass_fields__}
    (MODEL_DIR / "hmm_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    manifest = {
        "artifact": "pathsnap-hmm-viterbi-config",
        "format_version": 1,
        "decoder": "HMM + Viterbi",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": "hmm_config.json",
        "note": "Configuration for a deterministic decoder; not a fitted model checkpoint.",
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Exported HMM configuration and manifest to", MODEL_DIR)


if __name__ == "__main__":
    main()
