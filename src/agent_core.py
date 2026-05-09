"""Agent Core component for orchestrating LLM interactions and tool calling."""

from typing import Dict, List, Optional, Any
from src.models import AgentResponse, ToolCall, ToolResult
from src.todo_manager import ToDoManager
from src.memory_system import MemorySystem


class AgentCore:
    """Orchestrates LLM interactions with tool-calling capabilities.
    
    The AgentCore receives user input, analyzes intent using an LLM with function
    calling, executes appropriate tools (To-Do Manager, Memory System), and generates
    natural language responses incorporating tool results.
    """
    
    def __init__(
        self,
        openai_client,
        todo_manager: ToDoManager,
        memory_system: MemorySystem,
        system_prompt: Optional[str] = None
    ):
        """Initialize the AgentCore with LLM client and tools.
        
        Args:
            openai_client: OpenAI client instance for LLM API calls
            todo_manager: ToDoManager instance for task operations
            memory_system: MemorySystem instance for memory operations
            system_prompt: Optional custom system prompt (uses default if None)
        """
        self._client = openai_client
        self._todo_manager = todo_manager
        self._memory_system = memory_system
        self._system_prompt = system_prompt or self._get_default_system_prompt()
        
        from src.config import config
        self._model = config.openai_model
        
        # Session-based conversation context storage
        self._conversation_contexts: Dict[str, List[Dict]] = {}
        
        # Tool registry mapping tool names to handler methods
        self._tools = self._build_tool_registry()
    
    def _get_default_system_prompt(self) -> str:
        """Returns the default system prompt for the agent.
        
        Returns:
            str: System prompt defining agent personality and capabilities
        """
        from src.prompts import SYSTEM_PROMPT
        return SYSTEM_PROMPT

    
    def _build_tool_registry(self) -> List[Dict]:
        """Builds the tool registry with OpenAI function calling schemas.
        
        Returns:
            List[Dict]: List of tool definitions in OpenAI function calling format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_todo",
                    "description": "Creates a new to-do item with the given description. The item will have 'pending' status initially.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "The task description"
                            },
                            "status": {
                                "type": "string",
                                "description": "Optional status. Can be pending or completed."
                            }
                        },
                        "required": ["description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_todos",
                    "description": "Returns all to-do items. Use this when the user wants to see their tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_todo",
                    "description": "Updates a to-do item's description and/or status. Status must be one of: pending, completed, cancelled.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "todo_id": {
                                "type": "string",
                                "description": "The unique identifier of the to-do item"
                            },
                            "description": {
                                "type": "string",
                                "description": "New description (optional)"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "completed", "cancelled"],
                                "description": "New status (optional)"
                            }
                        },
                        "required": ["todo_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_todo",
                    "description": "Deletes a to-do item by its identifier.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "todo_id": {
                                "type": "string",
                                "description": "The unique identifier of the to-do item to delete"
                            }
                        },
                        "required": ["todo_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "store_memory",
                    "description": "Stores important information from the conversation for future recall. Use this for user preferences, facts about the user, or other information worth remembering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The information to remember"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tags for categorizing the memory (e.g., 'preference', 'personal', 'work')"
                            }
                        },
                        "required": ["content", "tags"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_memories",
                    "description": "Retrieves relevant memories from past conversations. Use this to get context about the user or previous interactions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant memories"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clear_memories",
                    "description": "Wipes the entire memory bank. Use this only when the user explicitly asks to 'clear all memories', 'delete all history', or 'forget everything'.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    
    def process_input(self, user_input: str, session_id: str = "default") -> AgentResponse:
        """Processes user input and returns an agent response.
        
        This method:
        1. Retrieves relevant memories based on user input
        2. Builds conversation context with memories
        3. Calls LLM with function calling to determine tool usage
        4. Executes any requested tools
        5. Generates final response incorporating tool results
        
        Args:
            user_input: The user's text input
            session_id: Session identifier for context management (default: "default")
            
        Returns:
            AgentResponse: Response containing text, tool calls, and success status
        """
        try:
            # Initialize session context if needed
            if session_id not in self._conversation_contexts:
                self._conversation_contexts[session_id] = []
            
            # Retrieve relevant memories automatically
            relevant_memories = self._memory_system.retrieve_memories(
                query=user_input,
                limit=3
            )
            
            # Build messages with system prompt, memories, context, and user input
            messages = self._build_messages(
                user_input=user_input,
                session_id=session_id,
                memories=relevant_memories
            )
            
            # Call LLM with function calling
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=self._tools,
                    tool_choice="auto"
                )
                assistant_message = response.choices[0].message
            except Exception as api_err:
                # If tool calling validation fails (common Groq/Llama quirk), 
                # fallback to a simple chat completion without tools
                print(f"Warning: Tool calling failed, falling back to chat: {api_err}")
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages
                )
                assistant_message = response.choices[0].message
            
            # Check if tools were called
            tool_calls = []
            tool_results = []
            
            if assistant_message.tool_calls:
                # Execute each tool call (proper API path)
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    import json
                    try:
                        tool_params = json.loads(tool_call.function.arguments)
                    except:
                        # Sometimes arguments are malformed, try to fix common Llama issues
                        raw_args = tool_call.function.arguments
                        if "{" not in raw_args:
                            continue # Skip malformed call
                        tool_params = json.loads(raw_args[raw_args.find("{"):raw_args.rfind("}")+1])
                    
                    # Create ToolCall object
                    tc = ToolCall(tool_name=tool_name, parameters=tool_params)
                    tool_calls.append(tc)
                    
                    # Execute tool
                    result = self.execute_tool(tool_name, tool_params)
                    tool_results.append(result)
                
                # Format final response with tool results
                final_text = self.format_response(
                    messages=messages,
                    assistant_message=assistant_message,
                    tool_results=tool_results
                )
            else:
                # No proper tool calls — check if the LLM embedded tool calls
                # as raw text (common Llama-3 behavior)
                raw_content = assistant_message.content or ""
                inline_calls = self._parse_inline_tool_calls(raw_content)
                
                if inline_calls:
                    # Execute the parsed inline tool calls
                    for tool_name, tool_params in inline_calls:
                        tc = ToolCall(tool_name=tool_name, parameters=tool_params)
                        tool_calls.append(tc)
                        result = self.execute_tool(tool_name, tool_params)
                        tool_results.append(result)
                    
                    # Generate a clean natural language response
                    final_text = self._format_response_fallback(tool_results)
                else:
                    final_text = raw_content
            
            # Update conversation context
            self._conversation_contexts[session_id].append({
                "role": "user",
                "content": user_input
            })
            self._conversation_contexts[session_id].append({
                "role": "assistant",
                "content": final_text
            })

            # --- Intent-Based Auto-Storage Fallback ---
            # If no formal tool was called, but the AI's response suggests it 
            # intended to remember something, we manually trigger a memory save.
            memory_keywords = ["i've remembered", "i will remember", "i've noted", "i've saved that", "noted your preference"]
            if not tool_calls and any(k in final_text.lower() for k in memory_keywords):
                # Use the user's input as the memory content
                print(f"[AETHER] Auto-storing memory based on intent: {user_input}")
                self._memory_system.store_memory(
                    content=user_input,
                    tags=["auto-captured", "preference"]
                )
            
            return AgentResponse(
                text=final_text,
                tool_calls=tool_calls,
                success=True,
                error=None
            )
            
        except Exception as e:
            # Handle any errors gracefully
            error_message = f"I encountered an error processing your request: {str(e)}"
            return AgentResponse(
                text=error_message,
                tool_calls=[],
                success=False,
                error=str(e)
            )
    
    def _parse_inline_tool_calls(self, text: str) -> list:
        """Parses tool calls embedded as raw text by the LLM.
        
        Llama-3 sometimes emits tool calls as raw strings instead of using
        the proper tool-calling API. Known formats:
          <function@update_todo>{"todo_id": "...", "status": "completed"}</function>
          <function=create_todo>{"description": "..."}</function>
          <function(update_todo)>{"todo_id": "..."}</function>
          <function(update_todo){"todo_id": "..."}</function>
        
        This method detects and parses those into (tool_name, params) tuples.
        
        Args:
            text: Raw response text from the LLM
            
        Returns:
            List of (tool_name, params_dict) tuples, empty if none found
        """
        import re
        import json
        
        results = []
        
        # Comprehensive patterns for all known Llama-3 inline function call formats
        patterns = [
            # <function@tool_name>{...}</function>  or  <function=tool_name>{...}</function>
            r'<function[@=](\w+)>\s*(\{.*?\})\s*</function>',
            # <function(tool_name)>{...}</function>  or  <function(tool_name){...}</function>
            r'<function\((\w+)\)>?\s*(\{.*?\})\s*(?:</function>)?',
            # <|python_tag|>tool_name.call({...})
            r'<\|python_tag\|>\s*(\w+)\.call\s*\(\s*(\{.*?\})\s*\)',
            # Bare tool_name({...}) pattern
            r'\b(create_todo|update_todo|delete_todo|list_todos|store_memory|recall_memories)\s*\(\s*(\{.*?\})\s*\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                tool_name = match[0]
                try:
                    params = json.loads(match[1])
                    results.append((tool_name, params))
                except json.JSONDecodeError:
                    continue
        
        return results
    
    def _build_messages(
        self,
        user_input: str,
        session_id: str,
        memories: List[Any]
    ) -> List[Dict]:
        """Builds the message list for LLM API call.
        
        Args:
            user_input: Current user input
            session_id: Session identifier
            memories: List of relevant Memory objects
            
        Returns:
            List[Dict]: Messages in OpenAI chat format
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        
        # ── Inject current task list so the agent can match by description ──
        try:
            current_todos = self._todo_manager.list_todos()
            if current_todos:
                task_context = "# YOUR CURRENT TASK LIST (use these IDs for update_todo / delete_todo):\n"
                for todo in current_todos:
                    task_context += f"- ID: \"{todo.id}\" | Description: \"{todo.description}\" | Status: {todo.status}\n"
                task_context += "\nWhen the user refers to a task by name, match it to one of the above and use the corresponding ID.\n"
            else:
                task_context = "# YOUR CURRENT TASK LIST: (empty — no tasks exist yet)\n"
            messages.append({"role": "system", "content": task_context})
        except Exception:
            pass  # If task list fails to load, continue without it
        
        # Add memory context if available
        if memories:
            memory_context = "Relevant information from past conversations:\n"
            for memory in memories:
                memory_context += f"- {memory.content}\n"
            messages.append({
                "role": "system",
                "content": memory_context
            })
        
        # Add conversation history
        context = self._conversation_contexts.get(session_id, [])
        messages.extend(context)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> ToolResult:
        """Executes a tool call and returns the result.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameters for the tool
            
        Returns:
            ToolResult: Result of the tool execution
        """
        try:
            if tool_name == "create_todo":
                item = self._todo_manager.create_todo(parameters["description"])
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result={
                        "id": item.id,
                        "description": item.description,
                        "status": item.status
                    },
                    error=None
                )
            
            elif tool_name == "list_todos":
                items = self._todo_manager.list_todos()
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=[
                        {
                            "id": item.id,
                            "description": item.description,
                            "status": item.status
                        }
                        for item in items
                    ],
                    error=None
                )
            
            elif tool_name == "update_todo":
                todo_id = parameters["todo_id"]
                description = parameters.get("description")
                status = parameters.get("status")
                
                item = self._todo_manager.update_todo(
                    todo_id=todo_id,
                    description=description,
                    status=status
                )
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result={
                        "id": item.id,
                        "description": item.description,
                        "status": item.status
                    },
                    error=None
                )
            
            elif tool_name == "delete_todo":
                self._todo_manager.delete_todo(parameters["todo_id"])
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result={"deleted": True},
                    error=None
                )
            
            elif tool_name == "store_memory":
                memory_id = self._memory_system.store_memory(
                    content=parameters["content"],
                    tags=parameters["tags"],
                    context={}
                )
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result={"memory_id": memory_id},
                    error=None
                )
            
            elif tool_name == "recall_memories":
                query = parameters["query"]
                limit = parameters.get("limit", 5)
                
                memories = self._memory_system.retrieve_memories(
                    query=query,
                    limit=limit
                )
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=[
                        {
                            "id": m.id,
                            "content": m.content,
                            "tags": m.tags,
                            "timestamp": m.timestamp.isoformat()
                        }
                        for m in memories
                    ],
                    error=None
                )

            elif tool_name == "clear_memories":
                success = self._memory_system.clear_all_memories()
                return ToolResult(
                    tool_name=tool_name,
                    success=success,
                    result={"cleared": success},
                    error=None if success else "Failed to clear memories"
                )
            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Unknown tool: {tool_name}"
                )
        
        except KeyError as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Item not found: {str(e)}"
            )
        except ValueError as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Invalid input: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool execution failed: {str(e)}"
            )
    
    def format_response(
        self,
        messages: List[Dict],
        assistant_message: Any,
        tool_results: List[ToolResult]
    ) -> str:
        """Formats tool results into a natural language response.
        
        This method calls the LLM again with tool results to generate
        a natural language response that incorporates the tool outcomes.
        
        Args:
            messages: The original conversation messages
            assistant_message: The original assistant message with tool calls
            tool_results: List of tool execution results
            
        Returns:
            str: Natural language response incorporating tool results
        """
        try:
            # Append the assistant's tool call message
            messages.append(assistant_message)

            
            # Add tool call results
            for i, tool_call in enumerate(assistant_message.tool_calls):
                result = tool_results[i]
                
                # Format tool result message
                if result.success:
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result.result)
                    }
                else:
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {result.error}"
                    }
                messages.append(tool_message)
            
            # Get final response from LLM
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            # Fallback to simple formatting if LLM call fails
            return self._format_response_fallback(tool_results)
    
    def _format_response_fallback(self, tool_results: List[ToolResult]) -> str:
        """Fallback method to format tool results without LLM.
        
        Args:
            tool_results: List of tool execution results
            
        Returns:
            str: Simple formatted response
        """
        if not tool_results:
            return "I processed your request."
        
        responses = []
        for result in tool_results:
            if result.success:
                if result.tool_name == "create_todo":
                    responses.append(f"Created task: {result.result['description']}")
                elif result.tool_name == "list_todos":
                    if result.result:
                        tasks = "\n".join([
                            f"- {item['description']} ({item['status']})"
                            for item in result.result
                        ])
                        responses.append(f"Your tasks:\n{tasks}")
                    else:
                        responses.append("You have no tasks.")
                elif result.tool_name == "update_todo":
                    responses.append(f"Updated task: {result.result['description']}")
                elif result.tool_name == "delete_todo":
                    responses.append("Task deleted.")
                elif result.tool_name == "store_memory":
                    responses.append("I'll remember that.")
                elif result.tool_name == "recall_memories":
                    if result.result:
                        memories = "\n".join([
                            f"- {mem['content']}"
                            for mem in result.result
                        ])
                        responses.append(f"I recall:\n{memories}")
                    else:
                        responses.append("I don't have any relevant memories.")
            else:
                responses.append(f"Error: {result.error}")
        
        return " ".join(responses)
