import json
from typing import Any, Dict, List, Optional, Union

# You already have this in llm.py
# from llm import call_llm


JsonType = Union[Dict[str, Any], List[Any]]


def load_json(path: str) -> JsonType:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _ensure_list_of_cases(data: JsonType) -> List[Dict[str, Any]]:
    """
    Your PSI file might be:
    - a list of case dicts: [ {...}, {...} ]
    - OR a dict containing a list somewhere (we try common keys)
    """
    if isinstance(data, list):
        # filter to dicts
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        # try common container keys
        for key in ["cases", "patients", "data", "items", "examples"]:
            if key in data and isinstance(data[key], list):
                return [x for x in data[key] if isinstance(x, dict)]

        # sometimes the dict itself is one case
        if "id" in data and isinstance(data.get("id"), str):
            return [data]

    raise ValueError("Could not interpret PSI JSON as a list of cases.")


def get_case_by_id(cases: List[Dict[str, Any]], case_id: str) -> Dict[str, Any]:
    for c in cases:
        if str(c.get("id", "")).strip() == str(case_id).strip():
            return c
    available = sorted({str(c.get("id")) for c in cases if c.get("id") is not None})
    raise KeyError(f"Case id '{case_id}' not found. Available ids include: {available[:20]} ...")


def _parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Tries to parse model output as JSON.
    Also strips common wrapping like ```json ... ```
    """
    t = text.strip()

    # remove fenced blocks if present
    if t.startswith("```"):
        # remove first fence line
        t = t.split("\n", 1)[1] if "\n" in t else ""
        # remove trailing fence
        if "```" in t:
            t = t.rsplit("```", 1)[0]
        t = t.strip()

    return json.loads(t)


def _validate_cactus_shape(obj: Dict[str, Any]) -> None:
    required_top = ["thought", "patterns", "intake_form", "cbt_technique", "cbt_plan"]
    for k in required_top:
        if k not in obj:
            raise ValueError(f"Missing top-level key: {k}")

    if not isinstance(obj["thought"], str) or not obj["thought"].strip():
        raise ValueError("thought must be a non-empty string")

    if not isinstance(obj["patterns"], list) or not obj["patterns"]:
        raise ValueError("patterns must be a non-empty list of strings")
    if any((not isinstance(x, str) or not x.strip()) for x in obj["patterns"]):
        raise ValueError("patterns must contain only non-empty strings")

    intake = obj["intake_form"]
    if not isinstance(intake, dict):
        raise ValueError("intake_form must be an object")

    if "client_info" not in intake or not isinstance(intake["client_info"], dict):
        raise ValueError("intake_form.client_info must be an object")

    ci_required = ["name", "age", "gender", "occupation", "education", "marital_status", "family_details"]
    for k in ci_required:
        if k not in intake["client_info"]:
            raise ValueError(f"Missing intake_form.client_info key: {k}")
        v = intake["client_info"][k]
        if k == "age":
            if not isinstance(v, (int, float)):
                raise ValueError("client_info.age must be a number")
        else:
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"client_info.{k} must be a non-empty string")

    list_fields = [
        ("presenting_problem", 3),
        ("past_history", 1),
        ("academic_occupational_functioning_level", 1),
    ]
    for field, min_len in list_fields:
        if field not in intake or not isinstance(intake[field], list) or len(intake[field]) < min_len:
            raise ValueError(f"intake_form.{field} must be a list with at least {min_len} items")
        if any((not isinstance(x, str) or not x.strip()) for x in intake[field]):
            raise ValueError(f"intake_form.{field} must contain only non-empty strings")

    if "reason_for_seeking_counseling" not in intake or not isinstance(intake["reason_for_seeking_counseling"], str) or not intake["reason_for_seeking_counseling"].strip():
        raise ValueError("intake_form.reason_for_seeking_counseling must be a non-empty string")

    if "social_support_system" not in intake or not isinstance(intake["social_support_system"], str) or not intake["social_support_system"].strip():
        raise ValueError("intake_form.social_support_system must be a non-empty string")

    if not isinstance(obj["cbt_technique"], str) or not obj["cbt_technique"].strip():
        raise ValueError("cbt_technique must be a non-empty string")

    plan = obj["cbt_plan"]
    if not isinstance(plan, dict):
        raise ValueError("cbt_plan must be an object")
    for k in ["1", "2", "3", "4", "5"]:
        if k not in plan or not isinstance(plan[k], str) or not plan[k].strip():
            raise ValueError("cbt_plan must contain non-empty strings for keys '1'..'5'")


def psi_case_to_cactus(
    psi_case: Dict[str, Any],
    *,
    call_llm_fn,  # pass llm.call_llm here
    system_prompt: str,
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Convert ONE PSI case dict -> CACTUS dict via LLM, with JSON parsing + validation + repair retries.
    """
    user_prompt = json.dumps(psi_case, ensure_ascii=False, indent=2)

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        if attempt == 0:
            response = call_llm_fn(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                model=model,
            )
        else:
            # Ask the model to fix its output using the error message.
            repair_user = (
                "Your previous output failed validation.\n"
                f"ERROR:\n{last_err}\n\n"
                "Return ONLY corrected JSON that matches the required schema exactly. No markdown."
            )
            response = call_llm_fn(
                system_prompt=system_prompt,
                user_prompt=repair_user + "\n\nORIGINAL PSI CASE:\n" + user_prompt,
                temperature=temperature,
                model=model,
            )

        try:
            obj = _parse_json_strict(response)
            _validate_cactus_shape(obj)
            return obj
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    raise ValueError(f"LLM output could not be validated after retries. Last error: {last_err}")


def convert_psi_file_case_id_to_cactus(
    psi_json_path: str,
    case_id: str,
    system_prompt_path: str,
    *,
    call_llm_fn,  # pass llm.call_llm here
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    High-level helper:
    - loads PSI json file
    - selects case by id
    - loads system prompt file
    - converts to CACTUS dict
    """
    data = load_json(psi_json_path)
    cases = _ensure_list_of_cases(data)
    psi_case = get_case_by_id(cases, case_id)

    system_prompt = load_text(system_prompt_path)

    cactus = psi_case_to_cactus(
        psi_case,
        call_llm_fn=call_llm_fn,
        system_prompt=system_prompt,
        temperature=temperature,
        model=model,
    )
    return cactus
