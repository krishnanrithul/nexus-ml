from typing import TypedDict, List, Annotated, Dict, Any, Optional
import operator
from ollama import Client

# Initialize the global Ollama client for all workers to use
client = Client(host='http://localhost:11434')

class FactoryState(TypedDict):
    """
    The shared memory for NexusML agents.
    """
    # File Paths
    raw_data_path: str
    cleaned_data_path: Optional[str]

    # Model Metadata
    model_results: Dict[str, Any]

    # Semantic Registry (what goes into LanceDB)
    report_chunks: List[str]

    # Agent Communication
    messages: Annotated[List[str], operator.add]  # appends across steps
    next_step: str

    # Error tracking
    errors: List[str]

    # Retry management
    retry_count: int  # incremented by manager on each retry, reset on success

    # Manager → Worker channel
    # When a worker fails, the manager diagnoses the error and writes a
    # targeted fix here. The worker reads this on retry and injects it
    # into its prompt so it knows exactly what to change.
    manager_diagnosis: str