import google.generativeai as genai
import google.generativeai.protos as protos
from typing import Any, Dict, List, Optional

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
    """Google Gemini 2.5 Flash implementation with function calling support"""

    SCHEMA_TYPE_MAP = {
        "string": protos.Type.STRING,
        "number": protos.Type.NUMBER,
        "integer": protos.Type.INTEGER,
        "boolean": protos.Type.BOOLEAN,
        "object": protos.Type.OBJECT,
        "array": protos.Type.ARRAY,
    }

    def __init__(self, config: Optional[LLMConfig] = None):
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

    async def generate_content(self, prompt: str) -> LLMResponse:
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

    def start_chat(
        self,
        history: Optional[List[ChatMessage]] = None,
        enable_function_calling: bool = False
    ) -> ChatSession:
        gemini_history = None
        if history:
            gemini_history = self._convert_history(history)

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
