
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
from dotenv import load_dotenv

from connectors.base_agent import BaseAgent, safe_extract_response_text
from connectors.agent_intelligence import (
    ConversationMemory,
    WorkspaceKnowledge,
    SharedContext,
    ProactiveAssistant
)

load_dotenv()


@dataclass
class OperationStats:
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    retry_count: int = 0
    total_pages_scraped: int = 0
    total_data_extracted: int = 0

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
        return f"{self.total_operations} ops, {success_rate:.1f}% success, {self.total_pages_scraped} pages scraped"


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
        self.agent_name = "scraper"
        self.verbose = verbose
        self.initialized = False

        self.session: Optional[ClientSession] = None
        self.stdio_context = None
        self.available_tools: List[Any] = []
        self.model: Optional[genai.GenerativeModel] = None

        self.memory = ConversationMemory()
        self.knowledge = WorkspaceKnowledge()
        self.shared_context = shared_context
        self.proactive = ProactiveAssistant('scraper', verbose)

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
        return """You are an expert web scraping and data extraction specialist with deep knowledge of web technologies, HTML/CSS parsing, and structured data extraction. Your mission is to help users extract valuable data from websites efficiently and reliably.

**Core Capabilities**:

**Page Scraping**:
- Scrape single web pages with full content
- Extract text, links, images, and metadata
- Handle JavaScript-rendered content (SPAs, React, Vue, etc.)
- Convert pages to clean Markdown format
- Extract structured data (JSON, tables, lists)
- Parse HTML and CSS selectors

**Website Crawling**:
- Crawl entire websites or specific sections
- Follow links automatically with depth control
- Respect robots.txt and crawl limits
- Filter URLs by patterns
- Handle pagination intelligently
- Discover sitemap and URL structures

**Structured Data Extraction**:
- Use AI to extract specific information
- Define extraction schemas (e.g., "get product name, price, rating")
- Extract lists and tables automatically
- Parse JSON-LD, microdata, and schema.org
- Clean and normalize extracted data
- Handle missing or malformed data gracefully

**Output Formats**:
- **Markdown**: Clean, readable text format
- **HTML**: Full HTML source with styles
- **JSON**: Structured data with defined schema
- **Text**: Plain text content only
- **Links**: Extract all links from page

**Best Practices**:

1. **Be Respectful & Ethical**:
   - Always respect robots.txt rules
   - Don't overload servers (reasonable delays)
   - Respect terms of service and privacy policies
   - Don't scrape personal or sensitive data without permission
   - Use appropriate user-agent strings
   - Avoid scraping login-protected content without authorization

2. **Handle Errors Gracefully**:
   - Validate URLs before scraping
   - Handle 404s, timeouts, and network errors
   - Retry transient failures intelligently
   - Report extraction failures clearly
   - Provide fallback options

3. **Optimize Performance**:
   - Cache repeated requests
   - Use efficient selectors
   - Limit crawl depth for large sites
   - Process data incrementally
   - Clean up resources after scraping

4. **Data Quality**:
   - Validate extracted data structure
   - Clean whitespace and formatting
   - Handle null/missing values
   - Normalize data types
   - Remove duplicates
   - Report data quality metrics

**Common Use Cases**:

**Product Data Extraction**:
```
Extract: product title, price, rating, availability, description
From: E-commerce product pages
Output: Structured JSON with fields
```

**Article Scraping**:
```
Extract: headline, author, publish date, body text, tags
From: News/blog pages
Output: Clean Markdown
```

**Directory Crawling**:
```
Crawl: Business directory website
Extract: Company name, address, phone, website
Output: CSV-compatible JSON
```

**Research Data Collection**:
```
Scrape: Academic papers, research data, citations
Extract: Title, authors, abstract, DOI, references
Output: Structured bibliography
```

**Price Monitoring**:
```
Extract: Current price, availability, shipping
Track: Changes over time
Output: Time-series data
```

**Troubleshooting Common Issues**:

- **Incomplete content**: Page uses JavaScript - enable JS rendering
- **Rate limiting**: Add delays between requests or use crawl settings
- **Access denied**: Site may block bots - check robots.txt and permissions
- **Malformed data**: Use AI extraction to handle inconsistent formats
- **Large site**: Use crawl limits and depth control

**YOU HAVE ACCESS TO FIRECRAWL TOOLS VIA FUNCTION CALLING**:
You have been provided with specialized web scraping tools that enable you to perform ALL scraping operations. These tools are your PRIMARY way of interacting with websites.

**When users ask you to scrape or extract data**, your FIRST action should be to call the appropriate tool. DO NOT say "I cannot scrape X" without attempting to use the available tools.

Examples:

User: "scrape the content from example.com"
✓ CORRECT: Call the scrape tool with the URL
✗ WRONG: "I cannot access websites"

User: "extract all product prices from this page"
✓ CORRECT: Call the extraction tool with appropriate schema
✗ WRONG: "Price extraction is not available"

User: "crawl the entire site and get all links"
✓ CORRECT: Call the crawl tool with link extraction
✗ WRONG: "I cannot crawl websites"

**If a tool call fails**, provide the error details and suggest solutions (check robots.txt, validate URL, adjust parameters). But ALWAYS TRY THE TOOL FIRST.

Remember: Web scraping is a powerful tool for data collection. Use it ethically, respect website owners, and always follow legal and ethical guidelines."""

    async def initialize(self) -> bool:
        """
        Initialize the scraper agent by connecting to Firecrawl MCP server

        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        try:
            if self.verbose:
                print(f"[SCRAPER AGENT] Initializing connection to Firecrawl MCP server")

            # Get Firecrawl API key
            firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
            if not firecrawl_api_key:
                print(f"[SCRAPER AGENT] Warning: FIRECRAWL_API_KEY not found in environment")
                print(f"[SCRAPER AGENT] Get your API key from https://firecrawl.dev")
                return False

            # Create MCP server parameters for Firecrawl
            # Uses official Firecrawl MCP via npx
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "firecrawl-mcp"],
                env={
                    "FIRECRAWL_API_KEY": firecrawl_api_key
                }
            )

            # Connect to MCP server
            await self._connect_to_mcp(server_params)

            # Load available tools
            await self._load_tools()

            # Initialize Gemini model with tools
            self._initialize_model()

            self.initialized = True

            # Feature #1: Prefetch metadata for faster operations (non-blocking, with timeout)
            try:
                await asyncio.wait_for(self._prefetch_metadata(), timeout=10.0)
            except asyncio.TimeoutError:
                if self.verbose:
                    print(f"[SCRAPER AGENT] Metadata prefetch timed out (continuing without cache)")
            except Exception as e:
                if self.verbose:
                    print(f"[SCRAPER AGENT] Metadata prefetch failed: {str(e)[:100]}")

            if self.verbose:
                print(f"[SCRAPER AGENT] Initialization complete. {len(self.available_tools)} tools available.")

            return True

        except Exception as e:
            error_msg = f"Failed to initialize Scraper agent: {str(e)}"
            print(f"[SCRAPER AGENT] {error_msg}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def _connect_to_mcp(self, server_params: StdioServerParameters):
        """
        Establish connection to MCP server

        Args:
            server_params: Server configuration parameters
                    session_logger: Optional session logger for tracking operations
        """
        self.stdio_context = stdio_client(server_params)
        stdio, write = await self.stdio_context.__aenter__()
        self.session = ClientSession(stdio, write)

        await self.session.__aenter__()
        await self.session.initialize()

    async def _load_tools(self):
        """
        Load available tools from MCP server

        Raises:
            RuntimeError: If no tools are available
        """
        tools_list = await self.session.list_tools()
        self.available_tools = tools_list.tools

        if not self.available_tools:
            raise RuntimeError("No tools available from Firecrawl MCP server")

    def _initialize_model(self):
        """
        Initialize the Gemini AI model with available tools
        """
        # Convert MCP tools to Gemini format
        gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]

        # Create model with configuration
        self.model = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            system_instruction=self.system_prompt,
            tools=gemini_tools
        )

    async def _prefetch_metadata(self):
        """
        Prefetch and cache scraping metadata for faster operations (Feature #1)

        This method caches common scraping configurations and patterns.
        The cache is persisted to the knowledge base with a 1-hour TTL.
        """
        try:
            # Check if we have valid cached metadata
            cached = self.knowledge.get_metadata_cache('scraper')
            if cached:
                self.metadata_cache = cached
                if self.verbose:
                    print(f"[SCRAPER AGENT] Loaded metadata from cache")
                return

            if self.verbose:
                print(f"[SCRAPER AGENT] Prefetching metadata...")

            # Store common configurations
            self.metadata_cache = {
                'output_formats': ['markdown', 'html', 'json', 'text', 'links'],
                'common_patterns': {
                    'product': ['name', 'price', 'rating', 'availability', 'description'],
                    'article': ['title', 'author', 'date', 'content', 'tags'],
                    'contact': ['name', 'email', 'phone', 'address', 'website']
                },
                'crawl_limits': {
                    'max_depth': 3,
                    'max_pages': 100
                },
                'fetched_at': asyncio.get_event_loop().time()
            }

            # Persist to knowledge base
            self.knowledge.save_metadata_cache('scraper', self.metadata_cache, ttl_seconds=3600)

            if self.verbose:
                print(f"[SCRAPER AGENT] Cached scraper metadata")

        except Exception as e:
            # Graceful degradation: If prefetch fails, continue without cache
            if self.verbose:
                print(f"[SCRAPER AGENT] Warning: Metadata prefetch failed: {e}")
            print(f"[SCRAPER AGENT] Continuing without metadata cache (operations may be slower)")

    async def get_capabilities(self) -> List[str]:
        """
        Get list of web scraping capabilities

        Returns:
            List of capability descriptions
        """
        return [
            "Scrape single web pages",
            "Crawl entire websites",
            "Extract structured data with AI",
            "Handle JavaScript-rendered content",
            "Convert pages to Markdown",
            "Extract specific data fields",
            "Follow links and pagination",
            "Respect robots.txt rules",
            "Export data in multiple formats",
            "Parse HTML and extract elements"
        ]

    async def execute(self, instruction: str) -> str:
        """
        Execute web scraping instruction

        Args:
            instruction: Natural language instruction from user

        Returns:
            str: Result of the scraping operation
                    session_logger: Optional session logger for tracking operations
        """
        if not self.initialized:
            return self._format_error(Exception("Scraper agent not initialized. Set FIRECRAWL_API_KEY environment variable."))

        try:
            # Resolve ambiguous references using conversation memory
            resolved_instruction = self._resolve_references(instruction)
            if resolved_instruction != instruction and self.verbose:
                print(f"[SCRAPER AGENT] Resolved instruction: {resolved_instruction}")

            # Check shared context from other agents
            context_from_other_agents = {}
            if self.shared_context:
                context_from_other_agents = self.shared_context.get_latest_context()

            if context_from_other_agents and self.verbose:
                print(f"[SCRAPER AGENT] Found context from other agents")

            # Start chat with instruction
            chat = self.model.start_chat(enable_automatic_function_calling=False)
            response = await chat.send_message_async(resolved_instruction)

            # Handle function calling loop with retry logic
            max_iterations = 15
            iteration = 0

            while iteration < max_iterations:
                parts = response.candidates[0].content.parts
                has_function_call = any(
                    hasattr(part, 'function_call') and part.function_call
                    for part in parts
                )

                if not has_function_call:
                    break

                # Extract function call
                function_call = next(
                    (part.function_call for part in parts
                     if hasattr(part, 'function_call') and part.function_call),
                    None
                )

                if not function_call:
                    break

                tool_name = function_call.name
                tool_args = self._deep_convert_proto_args(function_call.args)

                # Execute tool with retry logic (Feature #8)
                result = await self._execute_tool_with_retry(tool_name, tool_args, chat)

                if isinstance(result, dict) and result.get('isError'):
                    # Error occurred, send error response
                    response = result['response']
                else:
                    # Success, send result
                    response = result

                iteration += 1

            if iteration >= max_iterations:
                return "⚠️ Maximum scraping iterations reached. Task may be too complex."

            # Extract final text response
            final_response = safe_extract_response_text(response) if hasattr(response, 'text') else str(response)

            if self.verbose:
                print(f"\n[SCRAPER AGENT] Execution complete. {self.stats.get_summary()}")

            # Remember resources and add proactive suggestions
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
        """Resolve ambiguous references like 'it', 'that', 'this' using conversation memory"""
        ambiguous_terms = ['it', 'that', 'this', 'the page', 'the site', 'the url', 'the website']

        instruction_lower = instruction.lower()
        for term in ambiguous_terms:
            if term in instruction_lower:
                reference = self.memory.resolve_reference(term)
                if reference:
                    if self.verbose:
                        print(f"[SCRAPER AGENT] Resolved '{term}' → {reference}")
                    break

        return instruction

    def _remember_created_resources(self, response: str, instruction: str):
        """Remember URLs and scraped data for future reference"""
        # Extract URLs from response
        import re
        urls = re.findall(r'https?://[^\s]+', response)

        if urls:
            # Remember the most recent URL
            resource_id = urls[0]
            operation_type = self._infer_operation_type(instruction)

            self.memory.remember(
                operation_type,
                resource_id,
                {'url': resource_id, 'instruction': instruction}
            )

            # Share with other agents
            if self.shared_context:
                self.shared_context.share_resource(
                    'scraper',
                    'scraped_page',
                    resource_id,
                    {'url': resource_id}
                )

    def _infer_operation_type(self, instruction: str) -> str:
        """Infer the type of scraping operation from instruction"""
        instruction_lower = instruction.lower()

        if 'crawl' in instruction_lower or 'entire' in instruction_lower:
            return 'crawl_website'
        elif 'extract' in instruction_lower or 'get' in instruction_lower:
            return 'extract_data'
        elif 'scrape' in instruction_lower:
            return 'scrape_page'
        elif 'links' in instruction_lower:
            return 'extract_links'
        else:
            return 'general'

    async def _execute_tool_with_retry(self, tool_name: str, tool_args: Dict, chat: Any, retry_count: int = 0) -> Any:
        """
        Execute MCP tool with intelligent retry logic (Feature #8)

        Args:
            tool_name: Name of the tool to execute
            tool_args: Tool arguments
            chat: Active chat session
            retry_count: Current retry attempt number

        Returns:
            Response from tool execution or error dict
                    session_logger: Optional session logger for tracking operations
        """
        try:
            if self.verbose or retry_count > 0:
                retry_info = f" (retry {retry_count}/{RetryConfig.MAX_RETRIES})" if retry_count > 0 else ""
                print(f"\n[SCRAPER AGENT] Calling tool: {tool_name}{retry_info}")
                if self.verbose:
                    print(f"[SCRAPER AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

            # Call MCP tool
            result = await self.session.call_tool(tool_name, arguments=tool_args)

            # Extract result text
            result_text = ""
            if hasattr(result, 'content'):
                for item in result.content:
                    if hasattr(item, 'text'):
                        result_text += item.text + "\n"

            if self.verbose:
                print(f"[SCRAPER AGENT] Result: {result_text[:500]}")

            self.stats.record_success()
            self.stats.total_pages_scraped += 1

            # Send result back to model
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
                print(f"[SCRAPER AGENT] Error calling {tool_name}: {str(e)}")

            # Retry logic (Feature #8)
            if retry_count < RetryConfig.MAX_RETRIES:
                self.stats.record_retry()
                delay = min(RetryConfig.BASE_DELAY * (RetryConfig.EXPONENTIAL_BASE ** retry_count), RetryConfig.MAX_DELAY)

                if self.verbose:
                    print(f"[SCRAPER AGENT] Retrying in {delay:.1f}s...")

                await asyncio.sleep(delay)
                return await self._execute_tool_with_retry(tool_name, tool_args, chat, retry_count + 1)

            # Max retries reached, return error
            error_message = self._format_scraping_error(str(e), tool_name)

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

    def _format_scraping_error(self, error: str, tool_name: str) -> str:
        """Format scraping-specific errors with helpful context"""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return f"⏱️ Scraping timeout on {tool_name}. The website may be slow or rate limiting. Try again later."
        elif "access" in error_lower or "denied" in error_lower or "403" in error_lower:
            return f"🚫 Access denied for {tool_name}. The website may block automated scraping. Check robots.txt and permissions."
        elif "404" in error_lower or "not found" in error_lower:
            return f"🔍 Page not found. The URL may be incorrect or the page doesn't exist."
        elif "rate limit" in error_lower or "429" in error_lower:
            return f"⚠️ Rate limited. The website is blocking too many requests. Add delays or try later."
        else:
            return f"⚠️ Scraping error on {tool_name}: {error}"

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        """
        Validate if a scraping operation can be performed (Feature #14)

        Args:
            instruction: User instruction to validate

        Returns:
            Dict with validation results
                    session_logger: Optional session logger for tracking operations
        """
        result = {
            'valid': True,
            'missing': [],
            'warnings': [],
            'confidence': 1.0
        }

        instruction_lower = instruction.lower()

        # Check if URL is needed but not provided
        if 'http' not in instruction_lower:
            result['warnings'].append("No URL provided - may need clarification")
            result['confidence'] = 0.6

        # Warn about large crawls
        if 'crawl' in instruction_lower or 'entire' in instruction_lower:
            result['warnings'].append("Crawling entire websites can be slow and consume API credits")
            result['confidence'] = 0.8

        return result

    async def cleanup(self):
        """Cleanup scraper agent resources"""
        try:
            if self.verbose:
                print(f"\n[SCRAPER AGENT] Cleaning up. {self.stats.get_summary()}")

            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[SCRAPER AGENT] Error closing session: {e}")

        # IMPORTANT: The MCP stdio_client uses anyio which requires context managers
        # to be entered/exited in the same task. If cancelled, this may fail.
        try:
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except RuntimeError as e:
            # Specifically suppress "cancel scope in different task" errors
            # This happens when the agent is cancelled/timed out
            if "cancel scope" in str(e).lower() or "different task" in str(e).lower():
                # This is expected during cancellation, silently ignore
                pass
            elif self.verbose:
                print(f"[SCRAPER AGENT] Error closing stdio context: {e}")
        except Exception as e:
            if self.verbose:
                print(f"[SCRAPER AGENT] Error closing stdio context: {e}")

        self.initialized = False

    # Helper methods for Gemini integration

    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """Convert MCP tool schema to Gemini function declaration"""
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
        """Convert JSON schema to protobuf schema recursively"""
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
        """Recursively convert protobuf types to standard Python types"""
        type_str = str(type(value))

        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value

    def _format_error(self, error: Exception) -> str:
        """Format error message for user consumption"""
        return f"⚠️ Scraper agent error: {str(error)}"
