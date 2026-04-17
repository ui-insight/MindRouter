############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# canonical_schemas.py: Internal canonical request/response schemas
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Canonical request/response schemas for API translation.

These schemas serve as the internal representation that all incoming requests
are converted to, and from which backend-specific requests are generated.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content in a message."""
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"


class ImageDetail(str, Enum):
    """Image detail level for vision models."""
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class MessageRole(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ResponseFormatType(str, Enum):
    """Response format types."""
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


# Content Block Types
class TextContent(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str


class ImageUrlContent(BaseModel):
    """Image URL content block."""
    type: Literal["image_url"] = "image_url"
    image_url: Dict[str, Any]  # {"url": "...", "detail": "auto|low|high"}


class ImageBase64Content(BaseModel):
    """Base64 encoded image content block."""
    type: Literal["image_base64"] = "image_base64"
    data: str
    media_type: str = "image/png"


ContentBlock = Union[TextContent, ImageUrlContent, ImageBase64Content]


class CanonicalFunctionCall(BaseModel):
    """Function call within a tool call."""
    name: str
    arguments: str  # JSON string


class CanonicalToolCall(BaseModel):
    """A tool call from the assistant."""
    id: str
    type: str = "function"
    function: CanonicalFunctionCall


class CanonicalToolDefinition(BaseModel):
    """A tool definition provided in the request."""
    type: str = "function"
    function: Dict[str, Any]  # name, description, parameters


class CanonicalStreamToolCallDelta(BaseModel):
    """Delta for a tool call in a streaming response."""
    index: int
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[Dict[str, Any]] = None  # name, arguments (partial)


class CanonicalMessage(BaseModel):
    """Canonical message representation."""
    role: MessageRole
    content: Optional[Union[str, List[ContentBlock]]] = None
    reasoning: Optional[str] = Field(default=None, serialization_alias="reasoning_content")
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[CanonicalToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages

    def get_text_content(self) -> str:
        """Extract text content from message."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        text_parts = []
        for block in self.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return " ".join(text_parts)

    def has_images(self) -> bool:
        """Check if message contains images."""
        if self.content is None or isinstance(self.content, str):
            return False
        for block in self.content:
            if isinstance(block, (ImageUrlContent, ImageBase64Content)):
                return True
            if isinstance(block, dict) and block.get("type") in ("image_url", "image_base64"):
                return True
        return False


class ResponseFormat(BaseModel):
    """Response format specification."""
    type: ResponseFormatType = ResponseFormatType.TEXT
    json_schema: Optional[Dict[str, Any]] = None  # For json_schema type


class CanonicalChatRequest(BaseModel):
    """Canonical chat completion request."""
    # Required
    model: str
    messages: List[CanonicalMessage]

    # Optional parameters
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    seed: Optional[int] = None

    # Extended sampling parameters
    top_k: Optional[int] = Field(default=None, ge=1)
    repeat_penalty: Optional[float] = None  # vLLM calls this "repetition_penalty"
    min_p: Optional[float] = Field(default=None, ge=0, le=1)

    # Reasoning mode
    think: Optional[Union[bool, str]] = None  # bool for Qwen; "low"/"medium"/"high" for GPT-OSS
    reasoning_effort: Optional[str] = None  # "low", "medium", "high" (GPT-OSS)

    # Opaque backend-specific options (e.g. Ollama mirostat, tfs_z, num_ctx)
    backend_options: Optional[Dict[str, Any]] = None

    # Tool calling
    tools: Optional[List[CanonicalToolDefinition]] = None
    tool_choice: Optional[Any] = None

    # Structured output
    response_format: Optional[ResponseFormat] = None

    # Additional OpenAI-compatible fields
    n: int = 1  # Number of completions
    user: Optional[str] = None

    # MindRouter metadata (not sent to backend)
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None

    def requires_multimodal(self) -> bool:
        """Check if request requires multimodal capabilities."""
        return any(msg.has_images() for msg in self.messages)

    def requires_structured_output(self) -> bool:
        """Check if request requires structured output."""
        return (
            self.response_format is not None
            and self.response_format.type != ResponseFormatType.TEXT
        )

    def get_system_prompt(self) -> Optional[str]:
        """Extract system prompt from messages."""
        for msg in self.messages:
            if msg.role == MessageRole.SYSTEM:
                return msg.get_text_content()
        return None


class CanonicalCompletionRequest(BaseModel):
    """Canonical text completion request (legacy endpoint)."""
    model: str
    prompt: Union[str, List[str]]

    # Optional parameters
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    seed: Optional[int] = None
    suffix: Optional[str] = None
    echo: bool = False
    n: int = 1
    best_of: int = 1

    # Extended sampling parameters
    top_k: Optional[int] = Field(default=None, ge=1)
    repeat_penalty: Optional[float] = None  # vLLM calls this "repetition_penalty"
    min_p: Optional[float] = Field(default=None, ge=0, le=1)

    # Opaque backend-specific options (e.g. Ollama mirostat, tfs_z, num_ctx)
    backend_options: Optional[Dict[str, Any]] = None

    # Structured output
    response_format: Optional[ResponseFormat] = None

    # Thinking mode
    think: Optional[Union[bool, str]] = None  # bool for Qwen; "low"/"medium"/"high" for GPT-OSS

    # MindRouter metadata
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None

    def to_chat_request(self) -> CanonicalChatRequest:
        """Convert to chat request format."""
        prompt_text = self.prompt if isinstance(self.prompt, str) else self.prompt[0]
        return CanonicalChatRequest(
            model=self.model,
            messages=[CanonicalMessage(role=MessageRole.USER, content=prompt_text)],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            seed=self.seed,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            min_p=self.min_p,
            backend_options=self.backend_options,
            response_format=self.response_format,
            think=self.think,
            n=self.n,
            request_id=self.request_id,
            user_id=self.user_id,
            api_key_id=self.api_key_id,
        )


class CanonicalEmbeddingRequest(BaseModel):
    """Canonical embedding request."""
    model: str
    input: Union[str, List[str]]

    # Optional parameters
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None

    # MindRouter metadata
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None


# Response Schemas
class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    is_estimated: bool = False


class CanonicalChoice(BaseModel):
    """A single completion choice."""
    index: int = 0
    message: Optional[CanonicalMessage] = None
    text: Optional[str] = None  # For completion response
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class CanonicalStreamDelta(BaseModel):
    """Delta content for streaming responses."""
    role: Optional[MessageRole] = None
    content: Optional[str] = None
    reasoning: Optional[str] = Field(default=None, serialization_alias="reasoning_content")
    tool_calls: Optional[List[CanonicalStreamToolCallDelta]] = None


class CanonicalStreamChoice(BaseModel):
    """A streaming choice."""
    index: int = 0
    delta: CanonicalStreamDelta
    finish_reason: Optional[str] = None


class CanonicalChatResponse(BaseModel):
    """Canonical chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CanonicalChoice]
    usage: Optional[UsageInfo] = None

    # MindRouter additions
    backend_id: Optional[int] = None
    queue_delay_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None


class CanonicalStreamChunk(BaseModel):
    """Canonical streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[CanonicalStreamChoice]
    usage: Optional[UsageInfo] = None  # Only in final chunk


class CanonicalEmbeddingResponse(BaseModel):
    """Canonical embedding response."""
    object: str = "list"
    data: List[Dict[str, Any]]  # [{"object": "embedding", "embedding": [...], "index": 0}]
    model: str
    usage: UsageInfo


# Rerank / Score Schemas
class CanonicalRerankRequest(BaseModel):
    """Canonical rerank request."""
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: bool = True

    # MindRouter metadata
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None


class CanonicalRerankResult(BaseModel):
    """A single rerank result."""
    index: int
    relevance_score: float
    document: Optional[Dict[str, Any]] = None


class CanonicalRerankResponse(BaseModel):
    """Canonical rerank response."""
    id: str = ""
    model: str = ""
    results: List[CanonicalRerankResult] = []
    usage: UsageInfo = UsageInfo()


class CanonicalScoreRequest(BaseModel):
    """Canonical score request."""
    model: str
    text_1: str
    text_2: Union[str, List[str]]

    # MindRouter metadata
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None


class CanonicalScoreData(BaseModel):
    """A single score data point."""
    index: int
    score: float


class CanonicalScoreResponse(BaseModel):
    """Canonical score response."""
    id: str = ""
    object: str = "list"
    model: str = ""
    data: List[CanonicalScoreData] = []
    usage: UsageInfo = UsageInfo()


# Image Generation Schemas
class CanonicalImageRequest(BaseModel):
    """Canonical image generation request (OpenAI /v1/images/generations compatible)."""
    model: str
    prompt: str

    # Generation parameters
    n: int = 1  # Number of images
    size: str = "1024x1024"  # e.g. "1024x1024", "512x512", "1024x1792"
    quality: str = "standard"  # "standard" or "hd"
    style: Optional[str] = None  # "vivid" or "natural" (optional)
    response_format: Optional[str] = "url"  # "url" or "b64_json"

    # Extended parameters (FLUX-specific)
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None

    # MindRouter metadata
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None
    user: Optional[str] = None


class CanonicalImageData(BaseModel):
    """A single generated image."""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class CanonicalImageResponse(BaseModel):
    """Canonical image generation response."""
    created: int
    data: List[CanonicalImageData]

    # MindRouter additions
    backend_id: Optional[int] = None
    processing_time_ms: Optional[int] = None


class CanonicalErrorResponse(BaseModel):
    """Canonical error response."""
    error: Dict[str, Any]  # {"message": "...", "type": "...", "code": "..."}


# Model Information
class CanonicalModelInfo(BaseModel):
    """Model information for /v1/models listing."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mindrouter"

    # Extended info
    capabilities: Optional[Dict[str, bool]] = None  # vision, embeddings, structured_output
    backends: Optional[List[str]] = None  # List of backend names

    # Model details
    context_length: Optional[int] = None  # Effective context window (num_ctx injected)
    model_max_context: Optional[int] = None  # Architectural maximum
    parameter_count: Optional[str] = None  # e.g. "7B", "70B"
    quantization: Optional[str] = None  # e.g. "Q4_K_M", "FP16"
    family: Optional[str] = None  # e.g. "llama", "qwen2"


class CanonicalModelList(BaseModel):
    """List of available models."""
    object: str = "list"
    data: List[CanonicalModelInfo]
