from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from camel_agent import CamelCounselingSession, CounselorAgent, RESPONSE_PROMPT
from helpers import (
    cactus_to_intake_reason,
    format_dialogue,
    load_patients,
    load_text,
    next_phase,
    normalize_patient,
    parse_trust_score,
    parse_yes_no,
    pick_patient,
    render_template,
    trim_camel_history,
    trust_eval_interval,
)
from llm import call_llm_messages


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "Patient_PSi_CM_Dataset_Planning_Resistance.json"
DEFAULT_CACTUS_PATH = ROOT / "data" / "cactus_all_cases.json"
DEFAULT_CLIENT_PROMPT_PATH = ROOT / "prompts" / "client.txt"
DEFAULT_CRITIC_PROMPT_PATH = ROOT / "prompts" / "trust_critic.txt"
DEFAULT_MODERATOR_PROMPT_PATH = ROOT / "prompts" / "moderator.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a therapy session between a CAMEL therapist and a GPT client."
    )
    parser.add_argument("--case-id", default="2-1", help="PSI case id to simulate.")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum therapist-client turns.")
    parser.add_argument("--vllm-server", default="http://127.0.0.1:8000/v1", help="CAMEL vLLM server.")
    parser.add_argument("--camel-model-id", default="LangAGI-Lab/camel", help="Therapist model id.")
    parser.add_argument("--camel-temperature", type=float, default=0.7, help="Therapist temperature.")
    parser.add_argument("--client-model", default="gpt-4o-mini", help="Client model.")
    parser.add_argument("--critic-model", default="gpt-4o", help="Trust critic model.")
    parser.add_argument("--moderator-model", default="gpt-4o", help="Session moderator model.")
    parser.add_argument("--client-temperature", type=float, default=0.7, help="Client temperature.")
    parser.add_argument("--critic-temperature", type=float, default=0.0, help="Critic temperature.")
    parser.add_argument("--moderator-temperature", type=float, default=0.0, help="Moderator temperature.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the PSI patient dataset JSON file.",
    )
    parser.add_argument(
        "--cactus-path",
        type=Path,
        default=DEFAULT_CACTUS_PATH,
        help="Path to the precomputed CACTUS cases JSON file.",
    )
    parser.add_argument(
        "--client-prompt-path",
        type=Path,
        default=DEFAULT_CLIENT_PROMPT_PATH,
        help="Path to prompts/client.txt.",
    )
    parser.add_argument(
        "--critic-prompt-path",
        type=Path,
        default=DEFAULT_CRITIC_PROMPT_PATH,
        help="Path to prompts/trust_critic.txt.",
    )
    parser.add_argument(
        "--moderator-prompt-path",
        type=Path,
        default=DEFAULT_MODERATOR_PROMPT_PATH,
        help="Path to prompts/moderator.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the saved session JSON.",
    )
    parser.add_argument(
        "--no-print-turns",
        action="store_true",
        help="Disable per-turn stdout logging.",
    )
    return parser.parse_args()


def enrich_patient_from_cactus(patient: dict[str, Any], cactus_obj: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(patient)
    intake = cactus_obj.get("intake_form", {}) or {}
    client_info = intake.get("client_info", {}) or {}

    enriched["name"] = client_info.get("name", enriched.get("name", "Client"))

    past_history = intake.get("past_history", "")
    if isinstance(past_history, list):
        enriched["history"] = "\n".join(str(item) for item in past_history)
    elif past_history:
        enriched["history"] = str(past_history)

    thought = cactus_obj.get("thought")
    if thought:
        enriched["situation"] = thought

    return enriched


def load_cactus_case(cactus_path: Path, case_id: str) -> dict[str, Any]:
    data = json.loads(cactus_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected CACTUS data at {cactus_path} to be a JSON object keyed by case id.")
    try:
        case = data[str(case_id)]
    except KeyError as exc:
        raise KeyError(f"Case id '{case_id}' not found in {cactus_path}.") from exc
    if not isinstance(case, dict):
        raise ValueError(f"CACTUS entry for case id '{case_id}' must be a JSON object.")
    return case


def build_client_reply(
    convo: list[dict[str, str]],
    patient: dict[str, Any],
    client_template: str,
    client_model: str,
    client_temperature: float,
) -> str:
    client_system = render_template(client_template, patient)
    client_user = (
        "Conversation so far:\n"
        f"{format_dialogue(convo, last_n=24)}\n\n"
        "Respond as the client to the therapist's latest message."
    )
    return call_llm_messages(
        [
            {"role": "system", "content": client_system},
            {"role": "user", "content": client_user},
        ],
        temperature=client_temperature,
        model=client_model,
    )


def evaluate_openness(
    convo: list[dict[str, str]],
    critic_template: str,
    critic_model: str,
    critic_temperature: float,
) -> tuple[int | None, str]:
    critic_system = render_template(
        critic_template,
        {"dialogue_context": format_dialogue(convo, last_n=16)},
    )
    critic_text = call_llm_messages(
        [{"role": "system", "content": critic_system}],
        temperature=critic_temperature,
        model=critic_model,
    )
    return parse_trust_score(critic_text), critic_text


def should_end_session(
    convo: list[dict[str, str]],
    moderator_template: str,
    moderator_model: str,
    moderator_temperature: float,
) -> tuple[bool, str]:
    moderator_system = render_template(
        moderator_template,
        {"conversation": format_dialogue(convo, last_n=24)},
    )
    moderator_text = call_llm_messages(
        [{"role": "system", "content": moderator_system}],
        temperature=moderator_temperature,
        model=moderator_model,
    )
    end_flag = parse_yes_no(moderator_text)
    return (False if end_flag is None else end_flag), moderator_text


def print_turn_details(
    turn_id: int,
    therapist_text: str,
    client_text: str,
    critic_ran: bool,
    critic_text: str | None,
    moderator_end: bool,
    moderator_text: str,
) -> None:
    print(f"\n======= TURN {turn_id} =======")
    print(f"Therapist: {therapist_text}")
    print(f"Client: {client_text}")
    print(f"Critic ran: {'yes' if critic_ran else 'no'}")
    if critic_ran:
        print(f"Critic output: {critic_text}")
    print(f"Moderator end session: {'yes' if moderator_end else 'no'}")
    print(f"Moderator output: {moderator_text}")
    print("========================\n")


def simulate_session(args: argparse.Namespace) -> Path:
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    client_template = load_text(args.client_prompt_path)
    critic_template = load_text(args.critic_prompt_path)
    moderator_template = load_text(args.moderator_prompt_path)

    patients = load_patients(args.data_path)
    patient = normalize_patient(pick_patient(patients, patient_id=args.case_id))
    cactus_obj = load_cactus_case(args.cactus_path, args.case_id)
    patient = enrich_patient_from_cactus(patient, cactus_obj)

    sess = CamelCounselingSession(
        vllm_server=args.vllm_server,
        model_id=args.camel_model_id,
        temperature=args.camel_temperature,
    )
    intake_form, reason = cactus_to_intake_reason(sess, cactus_obj)

    interval = trust_eval_interval(patient.get("resistance_level"))
    phase = "trust_building"
    openness = 1
    trust_level = 1
    first_reply_generated = False

    convo: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []

    therapist_reply = "Hi, it's nice to meet you. What brings you to therapy today?"
    convo.append({"role": "assistant", "content": therapist_reply})

    for turn_id in range(1, args.max_turns + 1):
        patient["trust_level"] = trust_level
        patient["stage_therapy"] = phase

        client_text = build_client_reply(
            convo=convo,
            patient=patient,
            client_template=client_template,
            client_model=args.client_model,
            client_temperature=args.client_temperature,
        )
        convo.append({"role": "user", "content": client_text})

        should_eval = turn_id % interval == 0
        critic_text = None
        if should_eval:
            score, critic_text = evaluate_openness(
                convo=convo,
                critic_template=critic_template,
                critic_model=args.critic_model,
                critic_temperature=args.critic_temperature,
            )
            if score is not None:
                openness = score
            phase = next_phase(phase, openness)

        trust_level = openness

        end_flag, moderator_text = should_end_session(
            convo=convo,
            moderator_template=moderator_template,
            moderator_model=args.moderator_model,
            moderator_temperature=args.moderator_temperature,
        )

        turns.append(
            {
                "turn_id": turn_id,
                "phase_for_next_turn": phase,
                "openness": openness,
                "therapist": therapist_reply,
                "client": client_text,
                "critic_raw": critic_text,
                "moderator_raw": moderator_text,
                "end_session": end_flag,
            }
        )

        if not args.no_print_turns:
            print_turn_details(
                turn_id=turn_id,
                therapist_text=therapist_reply,
                client_text=client_text,
                critic_ran=should_eval,
                critic_text=critic_text,
                moderator_end=end_flag,
                moderator_text=moderator_text,
            )

        if end_flag:
            break

        if not first_reply_generated:
            sess.start(intake_form=intake_form, reason=reason, first_client_message=client_text)
            sess.history = trim_camel_history(sess.history, keep_last=25)

            counselor = CounselorAgent(
                sess.vllm_server,
                sess.model_id,
                sess.cbt_plan or "",
                RESPONSE_PROMPT,
            )
            therapist_reply = counselor.next_utterance(intake_form, reason, sess.history)
            sess.history.append({"role": "Counselor", "message": therapist_reply})
            sess.history = trim_camel_history(sess.history, keep_last=25)
            first_reply_generated = True
        else:
            sess.history = trim_camel_history(sess.history, keep_last=25)
            therapist_reply = sess.step(client_text)
            sess.history = trim_camel_history(sess.history, keep_last=25)

        convo.append({"role": "assistant", "content": therapist_reply})

    output = {
        "patient_id": str(patient.get("id", "")),
        "patient_name": str(patient.get("name", "")),
        "case_id": str(args.case_id),
        "therapist_model": args.camel_model_id,
        "client_model": args.client_model,
        "turns": turns,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = str(patient.get("name", "client")).replace("/", "-")
    output_path = args.output_dir / f"{safe_name}_{args.case_id}.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_path = simulate_session(args)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
