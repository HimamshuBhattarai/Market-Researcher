from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for all models: unknown fields are rejected instead of ignored."""

    model_config = ConfigDict(extra="forbid")