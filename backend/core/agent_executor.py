# backend/core/agent_executor.py
import json
import time
import re
from typing import Dict, Any, List, Optional
from backend.core.tool_registry import ToolRegistry
from backend.core.pipeline_loader import get_chat_pipeline
from backend.schemas.agent import ToolCall, ToolResult, AgentResponse


class AgentExecutor:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.chat_pipeline = None

    def _get_chat_pipeline(self):
        """Lazy load chat pipeline"""
        if self.chat_pipeline is None:
            self.chat_pipeline = get_chat_pipeline()
        return self.chat_pipeline

    def execute(
        self,
        query: str,
        available_tools: Optional[List[str]] = None,
        max_iterations: int = 3,
        temperature: float = 0.1,
    ) -> AgentResponse:
        """Execute agent query with tool calling"""
        start_time = time.time()

        # Filter available tools
        if available_tools is None:
            available_tools = self.tool_registry.list_tools()

        tool_descriptions = {
            name: self.tool_registry.get_tool(name)["description"]
            for name in available_tools
            if self.tool_registry.get_tool(name)
        }

        # Build system prompt with tools
        system_prompt = self._build_system_prompt(tool_descriptions)

        tool_calls = []
        reasoning_steps = []
        current_context = query

        for iteration in range(max_iterations):
            reasoning_steps.append(f"Iteration {iteration + 1}: Analyzing query")

            # Generate response with potential tool calls
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_context},
            ]

            response = self._call_llm(messages, temperature)

            # Parse tool calls from response
            tool_call = self._parse_tool_call(response)

            if tool_call is None:
                # No tool call, return final answer
                final_answer = response
                break

            # Execute tool
            tool_result = self._execute_tool(tool_call)
            tool_calls.append(tool_result)

            if tool_result.success:
                reasoning_steps.append(
                    f"Tool {tool_call.tool_name} executed successfully"
                )
                current_context += f"\n\nTool Result:\n{tool_result.result}"
            else:
                reasoning_steps.append(
                    f"Tool {tool_call.tool_name} failed: {tool_result.error}"
                )
                current_context += f"\n\nTool Error: {tool_result.error}"
        else:
            # Max iterations reached
            final_answer = "I apologize, but I couldn't complete the task within the allowed iterations."

        total_time = int((time.time() - start_time) * 1000)

        return AgentResponse(
            query=query,
            final_answer=final_answer,
            tool_calls=tool_calls,
            reasoning_steps=reasoning_steps,
            total_time_ms=total_time,
        )

    def _build_system_prompt(self, tool_descriptions: Dict[str, str]) -> str:
        """Build system prompt with available tools"""
        tools_text = "\n".join(
            [f"- {name}: {desc}" for name, desc in tool_descriptions.items()]
        )

        return f"""You are a helpful AI assistant with access to tools.

Available tools:
{tools_text}

When you need to use a tool, respond with this EXACT format:
TOOL_CALL: {{"tool_name": "tool_name", "parameters": {{"param1": "value1"}}}}

After receiving tool results, provide a final answer to the user.
If you don't need tools, respond directly without the TOOL_CALL format."""

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """Call LLM with messages"""
        try:
            pipeline = self._get_chat_pipeline()

            # Format messages into a single prompt
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

            response = pipeline(
                prompt,
                max_length=512,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=pipeline.tokenizer.eos_token_id,
            )

            return response[0]["generated_text"].split(prompt)[-1].strip()

        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    def _parse_tool_call(self, response: str) -> Optional[ToolCall]:
        """Parse tool call from LLM response"""
        try:
            # Look for TOOL_CALL: {...} pattern
            match = re.search(r"TOOL_CALL:\s*(\{.*?\})", response, re.DOTALL)
            if not match:
                return None

            tool_call_json = match.group(1)
            tool_call_data = json.loads(tool_call_json)

            return ToolCall(
                tool_name=tool_call_data["tool_name"],
                parameters=tool_call_data.get("parameters", {}),
            )

        except Exception as e:
            print(f"[Agent] Failed to parse tool call: {e}")
            return None

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call safely"""
        start_time = time.time()

        try:
            tool_info = self.tool_registry.get_tool(tool_call.tool_name)
            if not tool_info:
                return ToolResult(
                    tool_name=tool_call.tool_name,
                    success=False,
                    error=f"Tool '{tool_call.tool_name}' not found",
                    execution_time_ms=0,
                )

            # Validate parameters against tool signature
            func = tool_info["function"]
            sig = tool_info["signature"]

            # Bind parameters to function signature
            bound_args = sig.bind(**tool_call.parameters)
            bound_args.apply_defaults()

            # Execute function
            result = func(**bound_args.arguments)

            execution_time = int((time.time() - start_time) * 1000)

            return ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
