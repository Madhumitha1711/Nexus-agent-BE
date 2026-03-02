import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Literal

def merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    return {**existing, **new}

class AgentState(TypedDict):
    raw_input: str
    normalized_input: Optional[str]
    classification: Annotated[Dict[str, Any], merge_dicts]
    context_pool: Annotated[List[Dict[str, Any]], operator.add]
    root_cause_analysis: str
    confidence_score: float
    draft_response: Optional[str]
    final_output: Optional[str]
    resolution_status: Literal["auto", "hil", "pending"]
    internal_steps: Annotated[List[str], operator.add]