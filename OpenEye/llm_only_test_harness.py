import os
import json
import logging
import inspect
import time
import statistics
from openai import OpenAI
import sys

# Attempt to load tools from vr_agent without actually requiring anything
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from vr_agent.tools import _get_tools
except ImportError as e:
    print(f"Failed to import vr_agent tools: {e}")
    _get_tools = None

# Set up logging if needed
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def func_to_openai_tool(func):
    """Converts a standard Python function to an OpenAI tool JSON schema."""
    sig = inspect.signature(func)
    doc = func.__doc__ or f"Function {func.__name__}"
    
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        param_type = "string"
        # map python types to json schema types
        if param.annotation == float:
            param_type = "number"
        elif param.annotation == int:
            param_type = "integer"
        elif param.annotation == bool:
            param_type = "boolean"
        # If it's a typing type it can get complex but we'll guess reasonably
        elif str(param.annotation).startswith("typing.List"):
            param_type = "array"
            
        properties[name] = {"type": param_type}
        
        if param.default == inspect.Parameter.empty:
            required.append(name)
            
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

def get_agent_tools():
    """Returns a mocked list of agent tools formatted for OpenAI compatibility."""
    if not _get_tools:
        return []
        
    class MockObj:
        def call(self, *args, **kwargs): pass
        
    v_tools = _get_tools(MockObj(), MockObj(), MockObj(), MockObj(), MockObj(), MockObj())
    
    openai_tools = []
    for t in v_tools:
        openai_tools.append(func_to_openai_tool(t))
        
    # Also add standard top-level ones if missed
    return openai_tools

def main():
    print("======================================================")
    print(" Local LLM Test Harness (No Actions Executed)")
    print(" Make sure your local model is running via llama-server")
    print(" e.g. on http://100.100.219.101:8001/v1")
    print("======================================================")

    # Convert vr_agent tools:
    agent_tools = get_agent_tools()
    print(f"Loaded {len(agent_tools)} tool definitions from vr_agent")

    # Initialize the OpenAI client pointing to the local server
    try:
        # client = OpenAI(
        #     base_url="http://100.100.219.101:8001/v1",
        #     api_key="sk-no-key-required",
        # )
        # For using ngrok to run the llm on colab
        client = OpenAI(
            base_url="https://zippy-sarita-flabbier.ngrok-free.dev/v1", # Replace with your URL
            api_key="sk-no-key-required",
            default_headers={"ngrok-skip-browser-warning": "true"} # Bypasses the HTML warning
        )
        
        # Test connection by getting models
        models = client.models.list()
        model_name = models.data[0].id if models.data else "qwen3"
        print(f"Connected to local server! Using model: {model_name}")
    except Exception as e:
        print(f"Warning: Could not connect to local server at https://zippy-sarita-flabbier.ngrok-free.dev/v1")
        print(f"Error: {e}")
        print("Please ensure llama-server or your local LLM server is running.")
        model_name = "qwen3"

    print("\nYou can now type commands as you would for the VR agent.")
    print("The model's output will be printed, but no actions will be executed.")
    print("Type 'quit', 'exit', or 'stop' to abort.\n")

    # This is a sample system message representing what the agent might be told
    system_prompt = (
        "You are a helpful VR agent assistant with access to several tools. "
        "Your role is to help the user navigate their VR environment, control the headset/controllers, "
        "and perform actions via tool calls. "
        "ONLY use the available tools to perform actions. Return the correct tool call when asked."
    )
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    rt_times = []

    while True:
        try:
            user_input = input("\nYou (Type 'stop' to abort): ").strip()
            if not user_input:
                continue
                
            cmd = user_input.lower()
            if cmd in ["stop", "quit", "exit"]:
                print("Exiting test harness.")
                break

            messages.append({"role": "user", "content": user_input})
            
            # Start a multi-turn agent loop
            turn_count = 0
            while turn_count < 10:
                turn_count += 1
                if turn_count > 1:
                    print(f"\n[Continuing plan... Waiting for LLM step {turn_count}...]")
                else:
                    print("\n[Waiting for LLM response...]")
                
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.6,
                    "top_p": 0.95,
                }
                if agent_tools:
                    kwargs["tools"] = agent_tools
                    kwargs["tool_choice"] = "auto"

                start = time.perf_counter()
                response = client.chat.completions.create(**kwargs)
                elapsed = time.perf_counter() - start
                rt_times.append(elapsed)
                avg = statistics.mean(rt_times) if rt_times else 0.0
                print(f"\n[Benchmark] API call RTT: {elapsed:.3f}s (avg {avg:.3f}s over {len(rt_times)} calls)")
                message = response.choices[0].message
                content = message.content or ""
                
                print("\n[LLM Output]:")
                
                # Check for reasoning/thinking content if available (supported in some models)
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    print(f"--- Thinking ---\n{message.reasoning_content}\n----------------")
                    
                if content:
                    print(content)
                
                # Also mock tool calls if the model attempts them
                if message.tool_calls:
                    print("\n[LLM attempted to call tools (NOT EXECUTED)]:")
                    
                    # Store all tool calls generated by the assistant this turn
                    tool_calls_formatted = []
                    for tool_call in message.tool_calls:
                        tool_calls_formatted.append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                    
                    messages.append({
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": tool_calls_formatted
                    })
                    
                    # Print and provide mocked responses for each tool
                    for tool_call in message.tool_calls:
                        print(f"  Tool: {tool_call.function.name}")
                        print(f"  Args: {tool_call.function.arguments}")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": json.dumps({"status": f"Simulation only. Tool '{tool_call.function.name}' action completed successfully."})
                        })
                    
                    print("  -> Providing mock results to LLM and requesting next steps...")
                    # Let the inner loop continue so the LLM can generate the next action
                else:
                    # Append standard response and exit the multi-turn loop
                    messages.append({"role": "assistant", "content": content})
                    break
            
            if turn_count >= 10:
                print("\n[Warning: Reached maximum LLM turn limit (10). Breaking inner loop.]")
                
        except KeyboardInterrupt:
            print("\nExiting test harness.")
            break
        except Exception as e:
            print(f"\n[Error communicating with LLM]: {e}")

if __name__ == "__main__":
    main()
