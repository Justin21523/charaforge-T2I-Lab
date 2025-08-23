# backend/api/agent.py
from fastapi import APIRouter, HTTPException
from backend.schemas.agent import AgentRequest, AgentResponse
from backend.core.agent_executor import AgentExecutor

router = APIRouter()

# Global agent executor instance
agent_executor = AgentExecutor()


@router.post("/agent/act", response_model=AgentResponse)
async def agent_act(request: AgentRequest):
    """Execute agent query with tool calling"""
    try:
        response = agent_executor.execute(
            query=request.query,
            available_tools=request.tools,
            max_iterations=request.max_iterations,
            temperature=request.temperature,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/tools")
async def list_tools():
    """List available tools and their descriptions"""
    try:
        return {"tools": agent_executor.tool_registry.get_tool_descriptions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/reload")
async def reload_tools():
    """Reload tools from configuration"""
    try:
        agent_executor.tool_registry.reload_tools()
        return {"status": "success", "tools": agent_executor.tool_registry.list_tools()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
