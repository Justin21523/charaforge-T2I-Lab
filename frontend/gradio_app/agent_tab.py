# frontend/gradio_app/agent_tab.py
import gradio as gr
import requests
import json
from typing import List, Tuple


def call_agent_api(
    query: str, tools: str = "", max_iterations: int = 3
) -> Tuple[str, str]:
    """Call agent API and return formatted response"""
    try:
        # Parse tools list
        tool_list = (
            [t.strip() for t in tools.split(",") if t.strip()] if tools else None
        )

        # Make API request
        response = requests.post(
            "http://localhost:8000/api/v1/agent/act",
            json={
                "query": query,
                "tools": tool_list,
                "max_iterations": max_iterations,
                "temperature": 0.1,
            },
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        # Format response
        final_answer = data["final_answer"]

        # Format tool calls and reasoning
        details = f"**Query:** {data['query']}\n\n"
        details += f"**Total Time:** {data['total_time_ms']}ms\n\n"

        if data["reasoning_steps"]:
            details += "**Reasoning Steps:**\n"
            for step in data["reasoning_steps"]:
                details += f"• {step}\n"
            details += "\n"

        if data["tool_calls"]:
            details += "**Tool Executions:**\n"
            for tool_call in data["tool_calls"]:
                status = "✅" if tool_call["success"] else "❌"
                details += f"{status} **{tool_call['tool_name']}** ({tool_call['execution_time_ms']}ms)\n"
                if tool_call["success"]:
                    details += f"   Result: {tool_call['result']}\n"
                else:
                    details += f"   Error: {tool_call['error']}\n"
                details += "\n"

        return final_answer, details

    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}", ""
    except Exception as e:
        return f"Error: {str(e)}", ""


def get_available_tools() -> str:
    """Get list of available tools from API"""
    try:
        response = requests.get("http://localhost:8000/api/v1/agent/tools")
        response.raise_for_status()

        data = response.json()
        tools = data["tools"]

        tool_list = []
        for name, description in tools.items():
            tool_list.append(f"• **{name}**: {description}")

        return "\n".join(tool_list)

    except Exception as e:
        return f"Error loading tools: {str(e)}"


def create_agent_tab():
    """Create Gradio agent interface"""
    with gr.Tab("🤖 Agent"):
        gr.Markdown("# AI Agent with Tools")
        gr.Markdown("Ask questions and the agent will use available tools to help you.")

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Calculate the square root of 144, then search for information about mathematics",
                    lines=3,
                )

                with gr.Row():
                    tools_input = gr.Textbox(
                        label="Tools (comma-separated, leave empty for all)",
                        placeholder="calculator, web_search",
                        scale=2,
                    )
                    max_iter_input = gr.Slider(
                        label="Max Iterations",
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        scale=1,
                    )

                submit_btn = gr.Button("🚀 Execute", variant="primary")

            with gr.Column(scale=1):
                tools_display = gr.Markdown(
                    value=get_available_tools(), label="Available Tools"
                )
                reload_btn = gr.Button("🔄 Reload Tools")

        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Agent Answer", lines=5, interactive=False
                )

            with gr.Column():
                details_output = gr.Markdown(
                    label="Execution Details", value="", container=True
                )

        # Event handlers
        submit_btn.click(
            fn=call_agent_api,
            inputs=[query_input, tools_input, max_iter_input],
            outputs=[answer_output, details_output],
        )

        reload_btn.click(fn=get_available_tools, outputs=[tools_display])

        # Example queries
        gr.Examples(
            examples=[
                ["Calculate the area of a circle with radius 5", "", 2],
                ["Search for information about machine learning", "web_search", 1],
                [
                    "What's 15% of 1250, then search for tax information",
                    "calculator,web_search",
                    3,
                ],
                ["List files in the current directory", "list_files", 1],
                ["Calculate sin(pi/4) + cos(pi/3)", "calculator", 1],
            ],
            inputs=[query_input, tools_input, max_iter_input],
        )
