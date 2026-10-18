from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Question coming from the user."""

    question: str = Field(..., min_length=1, max_length=500)


class Source(BaseModel):
    """One retrieved chunk that supported the answer."""

    url: str
    snippet: str = Field(..., description="First ~200 chars of the chunk — shown in UI")
    full_context: str = Field(
        "",
        description="Full chunk text — for programmatic consumers of the REST API, not shown in UI",
    )


class ChatResponse(BaseModel):
    """Final answer with citations."""

    answer: str
    sources: list[Source]
