"""STUB operator-note interpreter.

Owned by Person A - this whole file will be replaced by the real LLM-backed
implementation. Only the signature below is the agreed contract.
"""

# Marker used by tests to skip LLM-dependent checks while the stub is in place.
# The real implementation should NOT define this.
IS_STUB = True


def interpret_notes(operator_notes: list[str], hours: list[dict],
                    battery: dict) -> list[dict]:
    """
    Returns one dict per note, in note_index order:
    {"note_index": int, "applies": bool, "directive_type": str,
     "structured_adjustment": dict|None, "explanation": str}
    """
    return [
        {
            "note_index": i,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Stub interpreter: note not interpreted.",
        }
        for i in range(len(operator_notes))
    ]
