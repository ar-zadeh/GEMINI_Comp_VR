import time
from vr_agent_qwen.executor import DirectMCPExecutor

def test():
    print("Initializing DirectMCPExecutor...")
    executor = DirectMCPExecutor()
    print("Calling inspect_surroundings...")
    try:
        res = executor.call("inspect_surroundings")
        if isinstance(res, str):
            print(f"Result (truncated): {res[:200]}")
        else:
            print("Result is not a string?!")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test()