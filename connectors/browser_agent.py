
import os
import asyncio
import time
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from connectors.base_agent import BaseAgent, safe_extract_response_text
from connectors.agent_intelligence import (
    ConversationMemory,
    WorkspaceKnowledge,
    SharedContext,
    ProactiveAssistant
)


@dataclass
class OperationStats:
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    retry_count: int = 0
    total_screenshots: int = 0

    def record_success(self):
        self.successful_operations += 1
        self.total_operations += 1

    def record_failure(self):
        self.failed_operations += 1
        self.total_operations += 1

    def record_retry(self):
        self.retry_count += 1

    def get_summary(self) -> str:
        if self.total_operations == 0:
            return "No operations yet"
        success_rate = (self.successful_operations / self.total_operations) * 100
        return f"{self.total_operations} ops, {success_rate:.1f}% success, {self.retry_count} retries"


@dataclass
class RetryConfig:
    MAX_RETRIES: int = 3
    BASE_DELAY: float = 1.0
    MAX_DELAY: float = 10.0
    EXPONENTIAL_BASE: float = 2.0


class Agent(BaseAgent):
    def __init__(self, verbose: bool = False, shared_context: Optional[SharedContext] = None,
        session_logger=None
    ):
        super().__init__()

        self.logger = session_logger
        self.agent_name = "browser"
        self.verbose = verbose
        self.initialized = False

        self.session: Optional[ClientSession] = None
        self.stdio_context = None
        self.available_tools: List[Any] = []
        self.model: Optional[genai.GenerativeModel] = None

        self.memory = ConversationMemory()
        self.knowledge = WorkspaceKnowledge()
        self.shared_context = shared_context
        self.proactive = ProactiveAssistant('browser', verbose)

        self.stats = OperationStats()
        self.metadata_cache = {}

        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """You are an expert web automation specialist with deep expertise in browser automation, web scraping, and data extraction. Your mission is to help users interact with websites programmatically, extract information, automate repetitive tasks, and test web applications.

**Core Capabilities**:

**Page Navigation**:
- Navigate to URLs and handle redirects
- Wait for page load and dynamic content
- Handle single-page applications (SPAs)
- Manage browser cookies and sessions
- Navigate browser history (back, forward, reload)

**Element Interaction**:
- Click buttons, links, and interactive elements
- Fill forms with data (text input, checkboxes, radio buttons, dropdowns)
- Upload and download files
- Scroll to elements and wait for visibility
- Handle iframes and shadow DOM
- Trigger keyboard and mouse events

**Data Extraction**:
- Extract text, HTML, and attributes from elements
- Scrape tables, lists, and structured data
- Parse JSON and API responses from network requests
- Extract metadata (titles, descriptions, og:tags)
- Handle pagination and infinite scroll
- Extract data from dynamic/AJAX-loaded content

**Screenshots & PDFs**:
- Capture full-page screenshots
- Take screenshots of specific elements
- Generate PDFs of web pages
- Customize viewport sizes for different devices

**Advanced Automation**:
- Execute custom JavaScript in page context
- Intercept and modify network requests
- Handle authentication flows (login, OAuth)
- Manage multiple pages/tabs
- Handle pop-ups, alerts, and dialogs
- Wait for custom conditions

**Best Practices**:

1. **Be Respectful**:
   - Respect robots.txt and terms of service
   - Add delays between requests to avoid overloading servers
   - Use realistic user-agent strings
   - Don't scrape personal or sensitive data without permission

2. **Handle Errors Gracefully**:
   - Wait for elements before interacting (don't assume instant load)
   - Handle timeouts and network errors
   - Verify elements exist before clicking/filling
   - Use try-catch for error-prone operations

3. **Be Efficient**:
   - Use selectors wisely (ID > class > complex CSS)
   - Wait for specific conditions, not arbitrary delays
   - Minimize screenshot/PDF generation (they're slow)
   - Close browsers and clean up resources

4. **Data Quality**:
   - Validate extracted data before returning
   - Handle missing or null values gracefully
   - Clean and format data appropriately
   - Report extraction success/failure clearly

**Common Patterns**:

**Login Flow**:
```
1. Navigate to login page
2. Fill username field
3. Fill password field
4. Click login button
5. Wait for redirect/dashboard
6. Verify login success
```

**Data Extraction**:
```
1. Navigate to target page
2. Wait for content to load
3. Select elements using CSS selectors
4. Extract text/attributes
5. Handle pagination if needed
6. Return structured data
```

**Form Automation**:
```
1. Navigate to form page
2. Wait for form to be visible
3. Fill each field sequentially
4. Handle dropdowns and checkboxes
5. Submit form
6. Wait for page response
```

**Debugging & Troubleshooting**:

- **Element not found**: Selector may be incorrect, or element loads dynamically
- **Timeout errors**: Page is slow, increase wait time or wait for specific elements
- **Stale element**: Page refreshed, re-query the element
- **Click not working**: Element may be hidden, overlapped, or outside viewport

**YOU HAVE ACCESS TO PLAYWRIGHT TOOLS VIA FUNCTION CALLING**:
You have been provided with specialized browser automation tools that enable you to perform ALL web automation operations. These tools are your PRIMARY way of interacting with web pages.

**When users ask you to do something**, your FIRST action should be to call the appropriate tool. DO NOT say "I cannot do X" or "I don't have access to Y" without attempting to use the available tools.

Examples:

User: "go to example.com"
✓ CORRECT: Call the navigate tool with URL
✗ WRONG: "I cannot navigate to websites"

User: "take a screenshot of the page"
✓ CORRECT: Call the screenshot tool
✗ WRONG: "Screenshot functionality is not available"

User: "click the submit button"
✓ CORRECT: Call the click tool with appropriate selector
✗ WRONG: "I cannot interact with page elements"

**If a tool call fails**, provide the error details and suggest solutions (better selectors, waiting for load, etc.). But ALWAYS TRY THE TOOL FIRST.

Remember: Be respectful, efficient, and accurate. Web automation is powerful - use it responsibly."""

    async def initialize(self) -> bool:
        try:
            if self.verbose:
                print(f"[BROWSER AGENT] Initializing connection to Playwright MCP server")

            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@playwright/mcp"],
                env=None
            )

            await self._connect_to_mcp(server_params)
            await self._load_tools()
            self._initialize_model()

            self.initialized = True
            await self._prefetch_metadata()

            if self.verbose:
                print(f"[BROWSER AGENT] Initialization complete. {len(self.available_tools)} tools available.")

            return True

        except Exception as e:
            error_msg = f"Failed to initialize Browser agent: {str(e)}"
            print(f"[BROWSER AGENT] {error_msg}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def _connect_to_mcp(self, server_params: StdioServerParameters):
        self.stdio_context = stdio_client(server_params)
        stdio, write = await self.stdio_context.__aenter__()
        self.session = ClientSession(stdio, write)

        await self.session.__aenter__()
        await self.session.initialize()

    async def _load_tools(self):
        tools_list = await self.session.list_tools()
        self.available_tools = tools_list.tools

        if not self.available_tools:
            raise RuntimeError("No tools available from Playwright MCP server")

    def _initialize_model(self):
        gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]

        self.model = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            system_instruction=self.system_prompt,
            tools=gemini_tools
        )

    async def _prefetch_metadata(self):
        try:
            cached = self.knowledge.get_metadata_cache('browser')
            if cached:
                self.metadata_cache = cached
                if self.verbose:
                    print(f"[BROWSER AGENT] Loaded metadata from cache")
                return

            if self.verbose:
                print(f"[BROWSER AGENT] Prefetching metadata...")

            self.metadata_cache = {
                'viewports': {
                    'desktop': {'width': 1920, 'height': 1080},
                    'tablet': {'width': 768, 'height': 1024},
                    'mobile': {'width': 375, 'height': 667}
                },
                'user_agents': {
                    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'firefox': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
                    'safari': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
                },
                'fetched_at': asyncio.get_event_loop().time()
            }

            self.knowledge.save_metadata_cache('browser', self.metadata_cache, ttl_seconds=3600)

            if self.verbose:
                print(f"[BROWSER AGENT] Cached browser metadata")

        except Exception as e:
            if self.verbose:
                print(f"[BROWSER AGENT] Warning: Metadata prefetch failed: {e}")
            print(f"[BROWSER AGENT] Continuing without metadata cache (operations may be slower)")

    async def get_capabilities(self) -> List[str]:
        return [
            "Navigate to websites and URLs",
            "Click buttons and interactive elements",
            "Fill forms and input fields",
            "Extract data and text from pages",
            "Take screenshots and generate PDFs",
            "Execute JavaScript in page context",
            "Handle authentication and cookies",
            "Automate complex workflows",
            "Scrape dynamic content",
            "Handle file uploads and downloads"
        ]

    async def execute(self, instruction: str) -> str:
        if not self.initialized:
            return self._format_error(Exception("Browser agent not initialized"))

        try:
            resolved_instruction = self._resolve_references(instruction)
            if resolved_instruction != instruction and self.verbose:
                print(f"[BROWSER AGENT] Resolved instruction: {resolved_instruction}")

            context_from_other_agents = {}
            if self.shared_context:
                context_from_other_agents = self.shared_context.get_latest_context()

            if context_from_other_agents and self.verbose:
                print(f"[BROWSER AGENT] Found context from other agents")

            chat = self.model.start_chat(enable_automatic_function_calling=False)
            response = await chat.send_message_async(resolved_instruction)

            max_iterations = 20
            iteration = 0

            while iteration < max_iterations:
                parts = response.candidates[0].content.parts
                has_function_call = any(
                    hasattr(part, 'function_call') and part.function_call
                    for part in parts
                )

                if not has_function_call:
                    break

                function_call = next(
                    (part.function_call for part in parts
                     if hasattr(part, 'function_call') and part.function_call),
                    None
                )

                if not function_call:
                    break

                tool_name = function_call.name
                tool_args = self._deep_convert_proto_args(function_call.args)

                result = await self._execute_tool_with_retry(tool_name, tool_args, chat)

                if isinstance(result, dict) and result.get('isError'):
                    response = result['response']
                else:
                    response = result

                iteration += 1

            if iteration >= max_iterations:
                return "⚠️ Maximum browser operation iterations reached. Task may be too complex."

            final_response = safe_extract_response_text(response) if hasattr(response, 'text') else str(response)

            if self.verbose:
                print(f"\n[BROWSER AGENT] Execution complete. {self.stats.get_summary()}")

            self._remember_created_resources(final_response, instruction)

            operation_type = self._infer_operation_type(instruction)
            suggestions = self.proactive.suggest_next_steps(operation_type, {})

            if suggestions:
                final_response += f"\n\n💡 {suggestions}"

            return final_response

        except Exception as e:
            self.stats.record_failure()
            return self._format_error(e)

    def _resolve_references(self, instruction: str) -> str:
        ambiguous_terms = ['it', 'that', 'this', 'the page', 'the site', 'the url']

        instruction_lower = instruction.lower()
        for term in ambiguous_terms:
            if term in instruction_lower:
                reference = self.memory.resolve_reference(term)
                if reference:
                    if self.verbose:
                        print(f"[BROWSER AGENT] Resolved '{term}' → {reference}")
                    break

        return instruction

    def _remember_created_resources(self, response: str, instruction: str):
        import re
        urls = re.findall(r'https?://[^\s]+', response)

        if urls:
            resource_id = urls[0]
            operation_type = self._infer_operation_type(instruction)

            self.memory.remember(
                operation_type,
                resource_id,
                {'url': resource_id, 'instruction': instruction}
            )

            if self.shared_context:
                self.shared_context.share_resource(
                    'browser',
                    'page',
                    resource_id,
                    {'url': resource_id}
                )

    def _infer_operation_type(self, instruction: str) -> str:
        instruction_lower = instruction.lower()

        if 'navigate' in instruction_lower or 'go to' in instruction_lower or 'visit' in instruction_lower:
            return 'navigate'
        elif 'click' in instruction_lower:
            return 'click'
        elif 'fill' in instruction_lower or 'enter' in instruction_lower or 'type' in instruction_lower:
            return 'fill_form'
        elif 'screenshot' in instruction_lower or 'capture' in instruction_lower:
            return 'screenshot'
        elif 'extract' in instruction_lower or 'scrape' in instruction_lower or 'get' in instruction_lower:
            return 'extract_data'
        else:
            return 'general'

    async def _execute_tool_with_retry(self, tool_name: str, tool_args: Dict, chat: Any, retry_count: int = 0) -> Any:
        try:
            if self.verbose or retry_count > 0:
                retry_info = f" (retry {retry_count}/{RetryConfig.MAX_RETRIES})" if retry_count > 0 else ""
                print(f"\n[BROWSER AGENT] Calling tool: {tool_name}{retry_info}")
                if self.verbose:
                    print(f"[BROWSER AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

            result = await self.session.call_tool(tool_name, arguments=tool_args)

            result_text = ""
            if hasattr(result, 'content'):
                for item in result.content:
                    if hasattr(item, 'text'):
                        result_text += item.text + "\n"

            if self.verbose:
                print(f"[BROWSER AGENT] Result: {result_text[:500]}")

            self.stats.record_success()

            return await chat.send_message_async(
                genai.protos.Content(
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": result_text}
                        )
                    )]
                )
            )

        except Exception as e:
            self.stats.record_failure()

            if self.verbose or retry_count > 0:
                print(f"[BROWSER AGENT] Error calling {tool_name}: {str(e)}")

            if retry_count < RetryConfig.MAX_RETRIES:
                self.stats.record_retry()
                delay = min(RetryConfig.BASE_DELAY * (RetryConfig.EXPONENTIAL_BASE ** retry_count), RetryConfig.MAX_DELAY)

                if self.verbose:
                    print(f"[BROWSER AGENT] Retrying in {delay:.1f}s...")

                await asyncio.sleep(delay)
                return await self._execute_tool_with_retry(tool_name, tool_args, chat, retry_count + 1)

            error_message = self._format_browser_error(str(e), tool_name)

            return {
                'isError': True,
                'response': await chat.send_message_async(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"error": error_message}
                            )
                        )]
                    )
                )
            }

    def _format_browser_error(self, error: str, tool_name: str) -> str:
        error_lower = error.lower()

        if "timeout" in error_lower:
            return f"⏱️ Browser timeout on {tool_name}. The page may be loading slowly or the element doesn't exist. Try increasing wait time or checking selectors."
        elif "selector" in error_lower or "element" in error_lower:
            return f"🔍 Element not found for {tool_name}. The CSS selector may be incorrect, or the element hasn't loaded yet. Verify selectors and wait for page load."
        elif "navigation" in error_lower:
            return f"🚫 Navigation failed. The URL may be incorrect or the site is unreachable. Check URL and network connection."
        else:
            return f"⚠️ Browser error on {tool_name}: {error}"

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        result = {
            'valid': True,
            'missing': [],
            'warnings': [],
            'confidence': 1.0
        }

        instruction_lower = instruction.lower()

        if any(word in instruction_lower for word in ['navigate', 'go to', 'visit', 'open']) and 'http' not in instruction_lower:
            result['warnings'].append("No URL provided - may need clarification")
            result['confidence'] = 0.7

        if any(word in instruction_lower for word in ['click', 'fill', 'extract']) and 'on' not in instruction_lower:
            result['warnings'].append("No element selector specified - may need discovery")
            result['confidence'] = 0.6

        return result

    async def cleanup(self):
        try:
            if self.verbose:
                print(f"\n[BROWSER AGENT] Cleaning up. {self.stats.get_summary()}")

            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[BROWSER AGENT] Error closing session: {e}")

        try:
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[BROWSER AGENT] Error closing stdio context: {e}")

        self.initialized = False

    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        parameters_schema = protos.Schema(type_=protos.Type.OBJECT)

        if hasattr(tool, 'inputSchema'):
            schema = tool.inputSchema
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    parameters_schema.properties[prop_name] = self._clean_schema(prop_schema)

            parameters_schema.required.extend(schema.get("required", []))

        return protos.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters=parameters_schema
        )

    def _clean_schema(self, schema: Dict) -> protos.Schema:
        schema_pb = protos.Schema()

        if "type" in schema:
            schema_pb.type_ = self.schema_type_map.get(
                schema["type"],
                protos.Type.TYPE_UNSPECIFIED
            )

        if "description" in schema:
            schema_pb.description = schema["description"]

        if "enum" in schema:
            schema_pb.enum.extend(schema["enum"])

        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema_pb.properties[prop_name] = self._clean_schema(prop_schema)

        if "items" in schema:
            schema_pb.items = self._clean_schema(schema["items"])

        return schema_pb

    def _deep_convert_proto_args(self, value: Any) -> Any:
        type_str = str(type(value))

        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value

    def _format_error(self, error: Exception) -> str:
        return f"⚠️ Browser agent error: {str(error)}"
