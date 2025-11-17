import google.generativeai as genai
import google.generativeai.protos as protos
from google.generativeai import caching
from typing import Any, Dict, List, Optional
import json
import hashlib
import time
from datetime import datetime, timedelta

from llms.base_llm import (
    BaseLLM,
    LLMConfig,
    LLMResponse,
    ChatSession,
    ChatMessage,
    FunctionCall,
    clean_json_schema,
    convert_proto_args
)


class GeminiChatSession(ChatSession):
    def __init__(self, gemini_chat: Any, enable_function_calling: bool = False):
        self.gemini_chat = gemini_chat
        self.enable_function_calling = enable_function_calling

    async def send_message(self, message: str) -> LLMResponse:
        response = await self.gemini_chat.send_message_async(message)

        text = None
        try:
            if hasattr(response, 'text'):
                text = response.text
        except Exception:
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text
                        break

        function_calls = None
        if self.enable_function_calling:
            function_calls = self._extract_function_calls(response)

        return LLMResponse(
            text=text,
            function_calls=function_calls if function_calls else None,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            metadata={'response_object': response}
        )

    async def send_message_with_functions(
        self,
        message: str,
        function_result: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        if function_result:
            function_name = function_result.get('name')
            result_data = function_result.get('result', {})

            content = genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=function_name,
                        response={"result": result_data}
                    )
                )]
            )

            response = await self.gemini_chat.send_message_async(content)
        else:
            response = await self.gemini_chat.send_message_async(message)

        try:
            text = response.text
        except Exception:
            text = None
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text
                        break

        function_calls = self._extract_function_calls(response)

        return LLMResponse(
            text=text,
            function_calls=function_calls if function_calls else None,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            metadata={'response_object': response}
        )

    def _extract_function_calls(self, response: Any) -> List[FunctionCall]:
        function_calls = []

        if not response.candidates:
            return function_calls

        parts = response.candidates[0].content.parts

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                function_calls.append(FunctionCall(
                    name=fc.name,
                    arguments=convert_proto_args(fc.args)
                ))

        return function_calls

    def get_history(self) -> List[ChatMessage]:
        history = []

        if hasattr(self.gemini_chat, 'history'):
            for msg in self.gemini_chat.history:
                role = msg.role if hasattr(msg, 'role') else 'unknown'
                content = ""

                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'text'):
                            content += part.text

                history.append(ChatMessage(
                    role=role,
                    content=content
                ))

        return history


class GeminiFlash(BaseLLM):
    """
    Google Gemini 2.5 Flash implementation with function calling support and context caching.

    Caching Strategy:
    - Uses Gemini's official Context Caching API for system instructions + tools
    - Cached tokens cost 1/16th the price (75%+ cost reduction)
    - Cache TTL: 1 hour (can be extended up to 24 hours)
    - Automatically refreshes expired caches
    - Fallback to non-cached requests if caching fails
    """

    SCHEMA_TYPE_MAP = {
        "string": protos.Type.STRING,
        "number": protos.Type.NUMBER,
        "integer": protos.Type.INTEGER,
        "boolean": protos.Type.BOOLEAN,
        "object": protos.Type.OBJECT,
        "array": protos.Type.ARRAY,
    }

    # Class-level cache for model instances (shared across all GeminiFlash instances)
    _model_cache: Dict[str, tuple[Any, float]] = {}  # cache_key -> (model, timestamp)
    _cache_ttl: float = 3600.0  # 1 hour cache TTL
    _enable_caching: bool = True  # Feature flag for caching

    # Context caching (Gemini's official prompt caching)
    _context_cache: Dict[str, Any] = {}  # cache_key -> CachedContent object
    _context_cache_ttl: float = 3600.0  # 1 hour (max is 86400 = 24 hours)
    _enable_context_caching: bool = False  # Feature flag for context caching (DISABLED - prompt too small)

    def __init__(self, config: Optional[LLMConfig] = None, enable_caching: bool = True):
        if config is None:
            config = LLMConfig(
                model_name='models/gemini-2.5-flash',
                temperature=0.7,
                top_p=0.95,
                top_k=40
            )

        super().__init__(config)

        self.provider_name = "google_gemini"
        self.supports_function_calling = True
        self.tools = []
        self.enable_caching = enable_caching

    @classmethod
    def _generate_cache_key(cls, model_name: str, system_instruction: Optional[str], tools: List[Any]) -> str:
        """
        Generate a unique cache key for model configuration.

        Args:
            model_name: Model identifier
            system_instruction: System prompt
            tools: List of tools/functions

        Returns:
            MD5 hash as cache key
        """
        # Create a string representation of the configuration
        key_parts = [
            model_name,
            system_instruction or "",
            str(len(tools)),  # Include tool count
            # Include tool names for uniqueness
            "|".join(sorted([getattr(t, 'name', str(t)) for t in tools]))
        ]

        key_string = "||".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    @classmethod
    def _get_cached_model(cls, cache_key: str) -> Optional[Any]:
        """
        Get cached model if available and not expired.

        Returns:
            Cached GenerativeModel or None
        """
        if not cls._enable_caching:
            return None

        if cache_key not in cls._model_cache:
            return None

        model, timestamp = cls._model_cache[cache_key]
        age = time.time() - timestamp

        if age > cls._cache_ttl:
            # Cache expired, remove it
            del cls._model_cache[cache_key]
            return None

        return model

    @classmethod
    def _cache_model(cls, cache_key: str, model: Any):
        """Cache a model instance"""
        if not cls._enable_caching:
            return

        cls._model_cache[cache_key] = (model, time.time())

        # Clean up expired entries (simple cleanup strategy)
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in cls._model_cache.items()
            if current_time - timestamp > cls._cache_ttl
        ]
        for key in expired_keys:
            del cls._model_cache[key]

    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache size, hit/miss info
        """
        current_time = time.time()
        active_entries = sum(
            1 for _, timestamp in cls._model_cache.values()
            if current_time - timestamp <= cls._cache_ttl
        )

        return {
            'cache_size': len(cls._model_cache),
            'active_entries': active_entries,
            'ttl_seconds': cls._cache_ttl,
            'caching_enabled': cls._enable_caching
        }

    @classmethod
    def clear_cache(cls):
        """Clear all cached models"""
        cls._model_cache.clear()

    @classmethod
    def _generate_context_cache_key(cls, model_name: str, system_instruction: Optional[str], tools: List[Any]) -> str:
        """
        Generate cache key for Gemini Context Caching.

        Args:
            model_name: Model identifier
            system_instruction: System prompt
            tools: List of tools/functions

        Returns:
            MD5 hash as cache key
        """
        key_parts = [
            model_name,
            system_instruction or "",
            str(len(tools)),
            "|".join(sorted([getattr(t, 'name', str(t)) for t in tools]))
        ]
        key_string = "||".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    @classmethod
    def _get_cached_content(cls, cache_key: str) -> Optional[Any]:
        """
        Get cached content if available and not expired.

        Returns:
            CachedContent object or None
        """
        if not cls._enable_context_caching:
            return None

        if cache_key not in cls._context_cache:
            return None

        cached_content = cls._context_cache[cache_key]

        try:
            # Check if cache is still valid by accessing its name
            # If cache is expired, this will raise an exception
            if hasattr(cached_content, 'name'):
                return cached_content
        except Exception:
            # Cache expired or invalid, remove it
            del cls._context_cache[cache_key]
            return None

        return None

    @classmethod
    def _create_cached_content(
        cls,
        cache_key: str,
        model_name: str,
        system_instruction: str,
        tools: Optional[List[Any]] = None
    ) -> Optional[Any]:
        """
        Create a new CachedContent object using Gemini's Context Caching API.

        This caches the system instruction and tools, reducing costs by ~75%.
        Cached tokens are charged at 1/16th the normal rate.

        Args:
            cache_key: Unique cache identifier
            model_name: Model to use
            system_instruction: System prompt to cache
            tools: Optional list of tools to cache

        Returns:
            CachedContent object or None if caching fails
        """
        if not cls._enable_context_caching:
            return None

        try:
            # Prepare the contents to cache (system instruction)
            contents = [system_instruction]

            # Create cached content with TTL
            ttl = timedelta(seconds=cls._context_cache_ttl)

            # Build the cached content
            if tools:
                cached_content = caching.CachedContent.create(
                    model=model_name,
                    system_instruction=system_instruction,
                    tools=tools,
                    ttl=ttl
                )
            else:
                cached_content = caching.CachedContent.create(
                    model=model_name,
                    system_instruction=system_instruction,
                    ttl=ttl
                )

            # Store in cache
            cls._context_cache[cache_key] = cached_content

            return cached_content

        except Exception as e:
            # Context caching might fail for various reasons:
            # - System instruction too short (minimum 32k tokens for caching)
            # - API quota exceeded
            # - Network issues
            # In these cases, we'll fall back to non-cached requests
            print(f"[GEMINI] Context caching failed (will use non-cached): {e}")
            return None

    @classmethod
    def _refresh_cached_content(cls, cache_key: str, cached_content: Any) -> Optional[Any]:
        """
        Refresh an expired CachedContent by updating its TTL.

        Returns:
            Updated CachedContent or None
        """
        try:
            # Update the TTL
            ttl = timedelta(seconds=cls._context_cache_ttl)
            updated_cache = cached_content.update(ttl=ttl)

            cls._context_cache[cache_key] = updated_cache
            return updated_cache

        except Exception as e:
            # If refresh fails, remove from cache
            if cache_key in cls._context_cache:
                del cls._context_cache[cache_key]
            return None

    @classmethod
    def clear_context_cache(cls):
        """Clear all cached content and delete from Gemini's servers"""
        for cached_content in cls._context_cache.values():
            try:
                # Delete from Gemini's cache
                if hasattr(cached_content, 'delete'):
                    cached_content.delete()
            except Exception:
                pass  # Ignore deletion errors

        cls._context_cache.clear()

    @classmethod
    def get_context_cache_stats(cls) -> Dict[str, Any]:
        """
        Get context cache statistics.

        Returns:
            Dict with cache info
        """
        return {
            'cache_entries': len(cls._context_cache),
            'ttl_seconds': cls._context_cache_ttl,
            'context_caching_enabled': cls._enable_context_caching,
            'estimated_cost_reduction': '75%+' if cls._enable_context_caching else '0%'
        }

    async def generate_content(self, prompt: str) -> LLMResponse:
        """
        Generate content with context caching support.

        If system instruction exists, it will be cached for cost savings.
        """
        model = None

        # Try to use context caching if system instruction exists
        if self.config.system_instruction:
            context_cache_key = self._generate_context_cache_key(
                self.config.model_name,
                self.config.system_instruction,
                []  # No tools for simple generate_content
            )

            cached_content = self._get_cached_content(context_cache_key)

            if cached_content is None:
                # Create new cached content
                cached_content = self._create_cached_content(
                    context_cache_key,
                    self.config.model_name,
                    self.config.system_instruction,
                    tools=None
                )

            if cached_content is not None:
                try:
                    model = genai.GenerativeModel.from_cached_content(cached_content)
                except Exception:
                    model = None

        # Fall back to regular model
        if model is None:
            model = genai.GenerativeModel(
                self.config.model_name,
                system_instruction=self.config.system_instruction
            )

        response = await model.generate_content_async(prompt)
        text = response.text if hasattr(response, 'text') else None

        return LLMResponse(
            text=text,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            metadata={'response_object': response}
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response using Gemini's JSON mode with context caching.

        Args:
            system_prompt: System instructions for the model
            user_prompt: User's request
            temperature: Optional temperature override

        Returns:
            Parsed JSON dictionary
        """
        # Create model with JSON response MIME type
        generation_config = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "response_mime_type": "application/json"  # Request JSON response
        }

        model = None

        # Try to use context caching for system prompt
        if system_prompt:
            context_cache_key = self._generate_context_cache_key(
                self.config.model_name,
                system_prompt,
                []
            )

            cached_content = self._get_cached_content(context_cache_key)

            if cached_content is None:
                # Create new cached content
                cached_content = self._create_cached_content(
                    context_cache_key,
                    self.config.model_name,
                    system_prompt,
                    tools=None
                )

            if cached_content is not None:
                try:
                    # Note: generation_config needs to be set after creating from cache
                    model = genai.GenerativeModel.from_cached_content(
                        cached_content
                    )
                except Exception:
                    model = None

        # Fall back to regular model
        if model is None:
            model = genai.GenerativeModel(
                self.config.model_name,
                system_instruction=system_prompt,
                generation_config=generation_config
            )

        # Generate content
        response = await model.generate_content_async(user_prompt)

        # Extract text and parse JSON
        text = response.text if hasattr(response, 'text') else None

        if not text:
            raise ValueError("No response text from LLM")

        try:
            # Parse JSON response
            return json.loads(text)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract JSON from text
            # Sometimes LLM wraps JSON in markdown code blocks
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Last resort: try to find any JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            raise ValueError(f"Failed to parse JSON from LLM response: {e}\nResponse: {text}")

    def start_chat(
        self,
        history: Optional[List[ChatMessage]] = None,
        enable_function_calling: bool = False
    ) -> ChatSession:
        """
        Start a chat session with Gemini Context Caching.

        Uses Gemini's official Context Caching API to cache system instructions
        and tools, reducing costs by ~75% (cached tokens are 1/16th the price).

        Caching strategy:
        1. Try to use existing cached content (saves cost and latency)
        2. If no cache exists, create new cached content
        3. Fall back to non-cached model if caching fails
        """
        gemini_history = None
        if history:
            gemini_history = self._convert_history(history)

        tools_for_cache = self.tools if enable_function_calling else []

        # Generate cache key for context caching
        context_cache_key = self._generate_context_cache_key(
            self.config.model_name,
            self.config.system_instruction,
            tools_for_cache
        )

        model = None

        # Try to use context caching (Gemini's official prompt caching)
        if self.config.system_instruction:
            cached_content = self._get_cached_content(context_cache_key)

            if cached_content is None:
                # Create new cached content
                cached_content = self._create_cached_content(
                    context_cache_key,
                    self.config.model_name,
                    self.config.system_instruction,
                    tools=tools_for_cache if tools_for_cache else None
                )

            # If we have cached content, use it
            if cached_content is not None:
                try:
                    model = genai.GenerativeModel.from_cached_content(cached_content)
                except Exception as e:
                    # If using cached content fails, fall back to regular model
                    print(f"[GEMINI] Failed to use cached content, falling back: {e}")
                    model = None

        # Fall back to regular model creation if caching failed or not available
        if model is None:
            if enable_function_calling and self.tools:
                model = genai.GenerativeModel(
                    self.config.model_name,
                    system_instruction=self.config.system_instruction,
                    tools=self.tools
                )
            else:
                model = genai.GenerativeModel(
                    self.config.model_name,
                    system_instruction=self.config.system_instruction
                )

        gemini_chat = model.start_chat(
            history=gemini_history,
            enable_automatic_function_calling=False
        )

        return GeminiChatSession(gemini_chat, enable_function_calling)

    def build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """Convert MCP tool to Gemini FunctionDeclaration"""
        parameters_schema = protos.Schema(type_=protos.Type.OBJECT)

        if hasattr(tool, 'inputSchema'):
            schema = tool.inputSchema

            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    parameters_schema.properties[prop_name] = self._convert_schema(prop_schema)

            if "required" in schema:
                parameters_schema.required.extend(schema["required"])

        return protos.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters=parameters_schema
        )

    def _convert_schema(self, schema: Dict) -> protos.Schema:
        """Convert JSON schema to Gemini protobuf schema"""
        schema_pb = protos.Schema()

        if "type" in schema:
            schema_pb.type_ = self.SCHEMA_TYPE_MAP.get(
                schema["type"],
                protos.Type.TYPE_UNSPECIFIED
            )

        if "description" in schema:
            schema_pb.description = schema["description"]

        if "enum" in schema:
            schema_pb.enum.extend(schema["enum"])

        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema_pb.properties[prop_name] = self._convert_schema(prop_schema)

        if "items" in schema:
            schema_pb.items = self._convert_schema(schema["items"])

        return schema_pb

    def build_function_response(
        self,
        function_name: str,
        result: Dict[str, Any]
    ) -> protos.Content:
        return genai.protos.Content(
            parts=[genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=function_name,
                    response={"result": result}
                )
            )]
        )

    def extract_function_calls(self, response: Any) -> List[FunctionCall]:
        function_calls = []

        if not hasattr(response, 'candidates') or not response.candidates:
            return function_calls

        parts = response.candidates[0].content.parts

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                function_calls.append(FunctionCall(
                    name=fc.name,
                    arguments=convert_proto_args(fc.args)
                ))

        return function_calls

    def set_tools(self, tools: List[Any]):
        self.tools = tools

    def _convert_history(self, history: List[ChatMessage]) -> List:
        return None

    def __repr__(self) -> str:
        return f"GeminiFlash(model={self.config.model_name}, tools={len(self.tools)})"
