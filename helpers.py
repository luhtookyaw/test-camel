import re
import json
from pathlib import Path
from camel_agent import CamelCounselingSession

PHASE_UPGRADE_AT = {
    "trust_building": 3,
    "case_conceptualization": 4,
    "solution_exploration": 999,
}

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def load_patients(data_path: Path) -> list:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "patients" in data:
        return data["patients"]
    raise ValueError("Unexpected patient JSON structure")

def pick_patient(patients: list, patient_id: str = None, index: int = None) -> dict:
    if patient_id is not None:
        for p in patients:
            if str(p.get("id")) == patient_id:
                return p
        raise ValueError(f"patient_id {patient_id} not found")

    if index is not None:
        if index < 0 or index >= len(patients):
            raise ValueError(f"index out of range: {index}")
        return patients[index]

    # default: first patient
    return patients[0]

def normalize_patient(p: dict) -> dict:
    core_beliefs = []
    core_beliefs += p.get("helpless_belief", []) or []
    core_beliefs += p.get("unlovable_belief", []) or []
    core_beliefs += p.get("worthless_belief", []) or []

    mapped = dict(p)
    mapped["core_beliefs"] = core_beliefs
    mapped["intermediate_beliefs"] = p.get("intermediate_belief", "") or ""

    mapped["resistance_emotions"] = p.get("resistance_emotion", "")
    mapped["resistance_monologue"] = p.get("resistance_internal_monologue", "")

    pt = p.get("type", [])
    mapped["patient_type_content"] = ", ".join(pt) if isinstance(pt, list) else str(pt)
    mapped["style_description"] = ""

    return mapped

def render_template(template: str, variables: dict) -> str:
    safe = {}
    for k, v in variables.items():
        if isinstance(v, (list, dict)):
            safe[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe[k] = "" if v is None else str(v)
    return template.format(**safe)

def parse_trust_score(text: str) -> int | None:
    m = re.search(r"\b([1-5])\b", text)
    return int(m.group(1)) if m else None

def parse_yes_no(text: str) -> bool | None:
    t = text.strip().upper()
    m = re.search(r"\b(YES|NO)\b", t)
    return (m.group(1) == "YES") if m else None

def format_dialogue(convo: list, last_n: int = 12) -> str:
    chunk = convo[-last_n:]
    out = []
    for msg in chunk:
        who = "Therapist" if msg["role"] == "assistant" else "Client"
        out.append(f"{who}: {msg['content']}")
    return "\n".join(out)

def next_phase(cur: str, openness: int) -> str:
    if cur == "trust_building" and openness >= PHASE_UPGRADE_AT["trust_building"]:
        return "case_conceptualization"
    if cur == "case_conceptualization" and openness >= PHASE_UPGRADE_AT["case_conceptualization"]:
        return "solution_exploration"
    return cur

def cactus_to_intake_reason(sess: CamelCounselingSession, cactus_obj: dict) -> tuple[str, str]:
    """
    Convert your cactus_obj intake_form -> CAMEL intake string + reason string.
    """
    intake_form = cactus_obj.get("intake_form", {}) or {}
    ci = intake_form.get("client_info", {}) or {}

    intake = sess.build_intake_form(
        name=str(ci.get("name", "")),
        age=str(ci.get("age", "")),
        gender=str(ci.get("gender", "")),
        occupation=str(ci.get("occupation", "")),
        education=str(ci.get("education", "")),
        marital_status=str(ci.get("marital_status", "")),
        family_details=str(ci.get("family_details", "")),
    )
    reason = intake_form.get("reason_for_seeking_counseling", "") or ""
    if not reason:
        reason = "The client seeks counseling support."
    return intake, reason

def trust_eval_interval(resistance_level: str) -> int:
    lvl = (resistance_level or "").strip().lower()
    if lvl == "beginner":
        return 2
    if lvl == "intermediate":
        return 4
    if lvl == "advanced":
        return 6
    # safe default
    return 2

def trim_camel_history(history, keep_last=10):
    """
    Keep greeting (first message) + last `keep_last` messages.
    history format: [{"role": "...", "message": "..."}]
    """
    if not history:
        return history
    head = history[:1]              # keep greeting
    tail = history[1:][-keep_last:] # keep recent context
    return head + tail

def print_last_turn(convo: list, turn_id: int):
    """
    Prints the most recent therapist + client exchange
    along with the turn number.
    """
    if len(convo) < 2:
        print(f"[Turn {turn_id}] Not enough turns yet.")
        return

    print(f"\n======= TURN {turn_id} =======")

    last_two = convo[-2:]
    for msg in last_two:
        role = "Therapist" if msg["role"] == "assistant" else "Client"
        print(f"{role}: {msg['content']}")

    print("========================\n")