from __future__ import annotations

import argparse
import json
import os
import re
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
from llm import call_llm, call_llm_messages
from src.alliance import C_ALLIANCE_SYSTEM_PROMPT, EXAMPLE_C_ALLIANCE
from src.therapist_skills import (
    CBT_SPECIFIC_FOCUS,
    CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
    CBT_SPECIFIC_STRATEGY,
    GEN_COLLABORATION,
    GEN_INTERPERSONAL,
    GEN_UNDERSTANDING,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "Patient_PSi_CM_Dataset_Planning_Resistance.json"
DEFAULT_CACTUS_PATH = ROOT / "data" / "cactus_all_cases.json"
DEFAULT_CLIENT_PROMPT_PATH = ROOT / "prompts" / "client.txt"
DEFAULT_CRITIC_PROMPT_PATH = ROOT / "prompts" / "trust_critic.txt"
DEFAULT_MODERATOR_PROMPT_PATH = ROOT / "prompts" / "moderator.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
THERAPIST_SKILL_PROMPTS = {
    "guided_discovery": CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
    "focus": CBT_SPECIFIC_FOCUS,
    "strategy": CBT_SPECIFIC_STRATEGY,
    "understanding": GEN_UNDERSTANDING,
    "interpersonal": GEN_INTERPERSONAL,
    "collaboration": GEN_COLLABORATION,
}


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
    parser.add_argument("--selector-model", default="gpt-4o", help="Model used to score therapist candidates.")
    parser.add_argument("--client-temperature", type=float, default=0.7, help="Client temperature.")
    parser.add_argument("--critic-temperature", type=float, default=0.0, help="Critic temperature.")
    parser.add_argument("--moderator-temperature", type=float, default=0.0, help="Moderator temperature.")
    parser.add_argument("--selector-temperature", type=float, default=0.0, help="Selector temperature.")
    parser.add_argument("--candidate-count", type=int, default=10, help="Number of CAMEL therapist candidates per turn.")
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


def format_candidate_lookahead_dialogue(
    convo: list[dict[str, str]],
    candidate_text: str,
    candidate_client_reply: str,
) -> str:
    candidate_convo = convo + [
        {"role": "assistant", "content": candidate_text},
        {"role": "user", "content": candidate_client_reply},
    ]
    return format_dialogue(candidate_convo, last_n=len(candidate_convo))


def parse_first_int(text: str, minimum: int = 0, maximum: int = 12) -> int | None:
    match = re.search(rf"\b([{minimum}-{maximum}])\b", text)
    if not match:
        return None
    return int(match.group(1))


def sum_numeric_scores(value: Any) -> int:
    total = 0
    if isinstance(value, dict):
        for nested in value.values():
            total += sum_numeric_scores(nested)
    elif isinstance(value, list):
        for nested in value:
            total += sum_numeric_scores(nested)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            total += int(stripped)
    elif isinstance(value, (int, float)):
        total += int(value)
    return total


def parse_alliance_output(raw_text: str) -> tuple[int, dict[str, Any] | None]:
    parsed_obj = None
    try:
        parsed_obj = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                parsed_obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed_obj = None

    if parsed_obj is not None:
        return sum_numeric_scores(parsed_obj), parsed_obj

    fallback_scores = [int(score) for score in re.findall(r'"score"\s*:\s*"(\d+)"', raw_text)]
    return sum(fallback_scores), None


def evaluate_alliance(
    conversation: str,
    model: str,
    temperature: float,
) -> tuple[int, str, dict[str, Any] | None]:
    prompt = C_ALLIANCE_SYSTEM_PROMPT.format(
        example=json.dumps(EXAMPLE_C_ALLIANCE, ensure_ascii=False, indent=2),
        conversation=conversation,
    )
    raw = call_llm(
        system_prompt=prompt,
        user_prompt="Return the alliance evaluation for the given conversation.",
        temperature=temperature,
        model=model,
    )
    score, parsed = parse_alliance_output(raw)
    return score, raw, parsed


def evaluate_therapist_skills(
    conversation: str,
    model: str,
    temperature: float,
) -> tuple[int, dict[str, dict[str, Any]]]:
    scores: dict[str, dict[str, Any]] = {}
    total = 0

    for skill_name, prompt_template in THERAPIST_SKILL_PROMPTS.items():
        prompt = prompt_template.format(conversation=conversation)
        raw = call_llm(
            system_prompt=prompt,
            user_prompt="Evaluate the therapist strictly and return the requested output format.",
            temperature=temperature,
            model=model,
        )
        score = parse_first_int(raw, minimum=0, maximum=6)
        score_value = 0 if score is None else score
        total += score_value
        scores[skill_name] = {
            "score": score_value,
            "raw": raw,
        }

    return total, scores


def generate_therapist_candidates(
    sess: CamelCounselingSession,
    intake_form: str,
    reason: str,
    history: list[dict[str, str]],
    candidate_count: int,
) -> list[str]:
    candidates: list[str] = []
    for _ in range(candidate_count):
        counselor = CounselorAgent(
            sess.vllm_server,
            sess.model_id,
            sess.cbt_plan or "",
            RESPONSE_PROMPT,
            temperature=sess.temperature,
            max_tokens=sess.max_tokens,
        )
        candidates.append(counselor.next_utterance(intake_form, reason, history))
    return candidates


def select_best_therapist_reply(
    convo: list[dict[str, str]],
    candidates: list[str],
    patient: dict[str, Any],
    client_template: str,
    client_model: str,
    client_temperature: float,
    model: str,
    temperature: float,
) -> tuple[str, list[dict[str, Any]]]:
    evaluations: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates, start=1):
        candidate_convo = convo + [{"role": "assistant", "content": candidate}]
        candidate_client_reply = build_client_reply(
            convo=candidate_convo,
            patient=patient,
            client_template=client_template,
            client_model=client_model,
            client_temperature=client_temperature,
        )
        conversation = format_candidate_lookahead_dialogue(
            convo=convo,
            candidate_text=candidate,
            candidate_client_reply=candidate_client_reply,
        )
        alliance_score, alliance_raw, alliance_parsed = evaluate_alliance(
            conversation=conversation,
            model=model,
            temperature=temperature,
        )
        skill_score, skill_details = evaluate_therapist_skills(
            conversation=conversation,
            model=model,
            temperature=temperature,
        )
        total_score = alliance_score + skill_score
        evaluations.append(
            {
                "candidate_id": idx,
                "response": candidate,
                "lookahead_client_reply": candidate_client_reply,
                "alliance_score": alliance_score,
                "alliance_raw": alliance_raw,
                "alliance_parsed": alliance_parsed,
                "therapist_skill_score": skill_score,
                "therapist_skill_details": skill_details,
                "total_score": total_score,
            }
        )

    best = max(evaluations, key=lambda item: item["total_score"])
    return best["response"], evaluations


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
    selected_reply: str | None = None,
    selected_score: int | None = None,
) -> None:
    print(f"\n======= TURN {turn_id} =======")
    print(f"Therapist: {therapist_text}")
    print(f"Client: {client_text}")
    print(f"Critic ran: {'yes' if critic_ran else 'no'}")
    if critic_ran:
        print(f"Critic output: {critic_text}")
    print(f"Moderator end session: {'yes' if moderator_end else 'no'}")
    print(f"Moderator output: {moderator_text}")
    if selected_reply is not None:
        print(f"Selected next therapist reply score: {selected_score}")
        print(f"Selected next therapist reply: {selected_reply}")
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
                "next_therapist_candidates": [],
                "selected_next_therapist": None,
            }
        )

        if end_flag:
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
            break

        if not first_reply_generated:
            sess.start(intake_form=intake_form, reason=reason, first_client_message=client_text)
            base_history = trim_camel_history(sess.history, keep_last=25)
            first_reply_generated = True
        else:
            trimmed_history = trim_camel_history(sess.history, keep_last=25)
            base_history = trimmed_history + [{"role": "Client", "message": client_text}]

        candidates = generate_therapist_candidates(
            sess=sess,
            intake_form=intake_form,
            reason=reason,
            history=base_history,
            candidate_count=args.candidate_count,
        )
        therapist_reply, candidate_evaluations = select_best_therapist_reply(
            convo=convo,
            candidates=candidates,
            patient=patient,
            client_template=client_template,
            client_model=args.client_model,
            client_temperature=args.client_temperature,
            model=args.selector_model,
            temperature=args.selector_temperature,
        )
        sess.history = trim_camel_history(
            base_history + [{"role": "Counselor", "message": therapist_reply}],
            keep_last=25,
        )
        turns[-1]["next_therapist_candidates"] = candidate_evaluations
        turns[-1]["selected_next_therapist"] = therapist_reply

        convo.append({"role": "assistant", "content": therapist_reply})

        output = {
            "patient_id": str(patient.get("id", "")),
            "patient_name": str(patient.get("name", "")),
            "case_id": str(args.case_id),
            "therapist_model": args.camel_model_id,
            "client_model": args.client_model,
            "selector_model": args.selector_model,
            "turns": turns,
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = str(patient.get("name", "client")).replace("/", "-")
        output_path = args.output_dir / f"{safe_name}_{args.case_id}.json"
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        if not args.no_print_turns:
            best_score = max(item["total_score"] for item in candidate_evaluations)
            print_turn_details(
                turn_id=turn_id,
                therapist_text=turns[-1]["therapist"],
                client_text=client_text,
                critic_ran=should_eval,
                critic_text=critic_text,
                moderator_end=end_flag,
                moderator_text=moderator_text,
                selected_reply=therapist_reply,
                selected_score=best_score,
            )

    output = {
        "patient_id": str(patient.get("id", "")),
        "patient_name": str(patient.get("name", "")),
        "case_id": str(args.case_id),
        "therapist_model": args.camel_model_id,
        "client_model": args.client_model,
        "selector_model": args.selector_model,
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
