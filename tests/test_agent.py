# tests/test_agent.py
import pytest
import asyncio
from backend.core.tool_registry import ToolRegistry
from backend.core.agent_executor import AgentExecutor
from backend.utils.calculator import calculate
from backend.utils.web_search import search_web


def test_calculator_tool():
    """Test calculator tool basic functionality"""
    assert calculate("2 + 2") == 4
    assert calculate("sqrt(16)") == 4.0
    assert calculate("sin(0)") == 0.0
    assert "Error" in calculate("1 / 0")
    assert "Error" in calculate("invalid_expression")


def test_web_search_tool():
    """Test web search tool (mock implementation)"""
    results = search_web("test query", max_results=3)
    assert len(results) == 3
    assert all("title" in result for result in results)
    assert all("url" in result for result in results)
    assert all("snippet" in result for result in results)


def test_tool_registry():
    """Test tool registry functionality"""
    registry = ToolRegistry("configs/agent.yaml")

    # Test tool listing
    tools = registry.list_tools()
    assert "calculator" in tools
    assert "web_search" in tools

    # Test tool retrieval
    calc_tool = registry.get_tool("calculator")
    assert calc_tool is not None
    assert calc_tool["function"] == calculate

    # Test descriptions
    descriptions = registry.get_tool_descriptions()
    assert "calculator" in descriptions
    assert len(descriptions["calculator"]) > 0


def test_agent_executor():
    """Test agent executor basic functionality"""
    executor = AgentExecutor()

    # Test simple calculation
    response = executor.execute(
        "Calculate 5 + 3", available_tools=["calculator"], max_iterations=2
    )

    assert response.final_answer is not None
    assert len(response.tool_calls) >= 0
    assert response.total_time_ms > 0


@pytest.mark.asyncio
async def test_agent_api_integration():
    """Test agent API endpoint integration"""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # Test tools endpoint
    response = client.get("/api/v1/agent/tools")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data

    # Test agent execution
    response = client.post(
        "/api/v1/agent/act",
        json={
            "query": "Calculate 2 + 2",
            "tools": ["calculator"],
            "max_iterations": 2,
            "temperature": 0.1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "final_answer" in data
    assert "tool_calls" in data
    assert "reasoning_steps" in data


if __name__ == "__main__":
    # Run basic tests
    test_calculator_tool()
    test_web_search_tool()
    test_tool_registry()
    test_agent_executor()
    print("All agent tests passed!")
