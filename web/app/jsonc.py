import re
import json


def loads(text: str) -> dict:
    """Strip // and /* */ comments, then parse as JSON."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return json.loads(text)


def dumps(obj: dict, **kwargs) -> str:
    return json.dumps(obj, indent=2, **kwargs)
