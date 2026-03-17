from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm import call_llm
from psi_to_cactus import convert_psi_file_all_cases_to_cactus


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT / "data" / "Patient_PSi_CM_Dataset_Planning_Resistance.json"
DEFAULT_PROMPT_PATH = ROOT / "prompts" / "psi_to_cactus_system.txt"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "cactus_all_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all PSI cases in a dataset to CACTUS format."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the PSI dataset JSON file.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the PSI->CACTUS system prompt file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the converted CACTUS JSON.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model used for the conversion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the conversion model.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any case fails conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = convert_psi_file_all_cases_to_cactus(
        psi_json_path=str(args.input_path),
        system_prompt_path=str(args.prompt_path),
        call_llm_fn=call_llm,
        temperature=args.temperature,
        model=args.model,
        skip_failed=not args.fail_fast,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(results)} converted cases to {args.output_path}")


if __name__ == "__main__":
    main()
