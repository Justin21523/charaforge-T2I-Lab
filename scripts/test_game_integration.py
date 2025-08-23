# scripts/test_game_integration.py
"""
Game Engine Integration Test Script
Tests the complete game flow from start to finish
"""

import requests
import json
import time

API_BASE = "http://localhost:8000/api/v1"


def test_complete_game_flow():
    """Test complete game flow"""
    print("🎮 Testing Complete Game Flow")

    # 1. Get available personas
    print("\n1. Getting personas...")
    response = requests.get(f"{API_BASE}/game/personas")
    personas = response.json()["personas"]
    print(f"   Available personas: {list(personas.keys())}")

    # 2. Start new game
    print("\n2. Starting new game...")
    game_data = {
        "persona": "wise_mentor",
        "setting": "modern",
        "difficulty": "normal",
        "player_name": "整合測試玩家",
    }

    response = requests.post(f"{API_BASE}/game/new", json=game_data)
    game_state = response.json()
    session_id = game_state["session_id"]
    print(f"   Session ID: {session_id}")
    print(f"   Scene: {game_state['scene']['title']}")
    print(f"   Choices: {len(game_state['choices'])}")

    # 3. Take several actions
    actions = ["greet", "observe", "ask_question"]
    for i, action in enumerate(actions, 1):
        print(f"\n3.{i} Taking action: {action}")

        action_data = {
            "session_id": session_id,
            "action": action,
            "message": f"這是第{i}個測試行動",
        }

        response = requests.post(f"{API_BASE}/game/step", json=action_data)
        if response.status_code == 200:
            game_state = response.json()
            print(f"   Turn: {game_state['turn_number']}")
            print(f"   Health: {game_state['player_state']['health']}")
            print(f"   Energy: {game_state['player_state']['energy']}")
        else:
            print(f"   Error: {response.status_code}")

        time.sleep(1)  # Prevent overwhelming the API

    # 4. Save game
    print("\n4. Saving game...")
    save_data = {"session_id": session_id, "save_name": "integration_test_save"}

    response = requests.post(f"{API_BASE}/game/save", json=save_data)
    if response.status_code == 200:
        print("   Game saved successfully")
    else:
        print(f"   Save failed: {response.status_code}")

    # 5. List saves
    print("\n5. Listing saves...")
    response = requests.get(f"{API_BASE}/game/saves")
    saves = response.json()["saves"]
    print(f"   Available saves: {saves}")

    # 6. Load game
    print("\n6. Loading game...")
    load_data = {"save_name": "integration_test_save"}

    response = requests.post(f"{API_BASE}/game/load", json=load_data)
    if response.status_code == 200:
        loaded_state = response.json()
        print(f"   Loaded session: {loaded_state['session_id']}")
        print(f"   Turn number: {loaded_state['turn_number']}")
    else:
        print(f"   Load failed: {response.status_code}")

    print("\n✅ Complete game flow test finished!")


if __name__ == "__main__":
    try:
        test_complete_game_flow()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
