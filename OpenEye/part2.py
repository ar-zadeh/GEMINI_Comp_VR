# ============================================================================
# AGENT (Multi-Model Orchestrator)
# ============================================================================

class GeminiAgent:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key: raise ValueError("GEMINI_API_KEY not set")
        
        # Clients for different models
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        self.logger = get_logger()
        
        # Initialize Core Components
        self.executor = DirectMCPExecutor()
        self.grounder = VisualGrounder(self.client, LOG_DIR)
        self.planner = ActionPlanner(self.client)
        self.verifier = Verifier(self.client)
        self.describer = Describer(self.client)
        self.white_cane = WhiteCaneAssistant(self.client, self.executor, LOG_DIR)
        self.live_white_cane = LiveWhiteCaneAgent(self.client, self.executor, LOG_DIR)
        
        self.chat_history = [] # Store conversation
        
        if ObjectTracker:
            self.tracker = ObjectTracker(LOG_DIR)
        else:
            self.tracker = None
        
        # Tools (for execution phase)
        self.tools = _get_tools(self.executor, self.grounder, self.tracker, self.white_cane, self.describer, self)
        # Create a mapping for manual execution from plan
        self.tool_map = {t.__name__: t for t in self.tools}
        self.tool_map["describe_view"] = self._describe_view_tool
        self.tool_map["verify_action"] = self._verify_action_tool
        
        # Start bridge immediately
        self.executor.call("start_vr_bridge")

        # Initialize keyboard controller (uses termios — works in WSL/Linux)
        try:
            from keyboard_controller import KeyboardVRController
            self.keyboard_ctrl = KeyboardVRController(self.executor.module)
            print("Keyboard VR control available (Default: Trackpad Mode). Type ` (backtick) at the prompt to toggle.")
        except ImportError:
            self.keyboard_ctrl = None
        
    def _describe_view_tool(self, question: str):
        """Tool wrapper for description model."""
        print("Capturing image for description...")
        res = self.executor.call("inspect_surroundings")
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            return self.describer.describe(img_bytes, question)
        except Exception as e:
            return f"Description failed: {e}"

    def _verify_action_tool(self, action_description: str):
        """Tool wrapper for verification model."""
        print("Capturing image for verification...")
        res = self.executor.call("inspect_surroundings")
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            return self.verifier.verify(img_bytes, action_description)
        except Exception as e:
            return f"Verification failed: {e}"

    def run(self, user_input: str):
        self.logger.info(f"User: {user_input}")
        self.chat_history.append({"role": "user", "content": user_input})
        print("\nAgent (Planner) is thinking...")
        
        # 1. PLANNING PHASE (Gemini 3 Flash)
        plan = self.planner.create_plan(user_input)
        
        if not plan:
            print("Failed to generate a plan.")
            return

        print(f"\nGenerated Plan ({len(plan)} steps):")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step.tool}: {step.description}")

        # 2. EXECUTION PHASE
        print("\nExecuting Plan...")
        for step in plan:
            print(f"\n>> Step: {step.tool}({step.args})")
            func = self.tool_map.get(step.tool)
            
            if func:
                try:
                    # Execute tool
                    result = func(**step.args)
                    print(f"Result: {str(result)[:200]}...")
                    self.chat_history.append({"role": "agent", "content": f"Executed {step.tool}: {result}"})
                    self.logger.info(f"Step '{step.tool}' Result: {result}")
                except Exception as e:
                    print(f"Execution Error: {e}")
                    self.logger.error(f"Execution Error in {step.tool}: {e}")
                    break
            else:
                print(f"Error: Unknown tool '{step.tool}'")



    def print_status(self):
        """Print current agent status."""
        print(f"\n--- Status (v2 Multi-Model) ---")
        print(f"Planner: {MODEL_PLANNER}")
        print(f"Grounding: {MODEL_GROUNDING}")
        print(f"Verification: {MODEL_VERIFICATION}")
        print(f"Description: {MODEL_DESCRIPTION}")
        print(f"Log Dir: {LOG_DIR}")        
        try:
            status = self.executor.call("get_connection_status")
            print(f"VR Bridge: {status}")
        except Exception as e:
            print(f"VR Bridge: Error getting status ({e})")
        print("--------------")



    def handle_direct_command(self, user_input: str):
        """
        Parses and executes a direct command in the format ((function arg1 arg2 ...))
        Arguments are automatically converted to int/float/bool/None if possible.
        """
        try:
            # Strip (( and ))
            content = user_input[2:-2].strip()
            if not content:
                print("Empty direct command.")
                return

            # Parse with shlex (handles quoted strings)
            parts = shlex.split(content)
            func_name = parts[0]
            raw_args = parts[1:]
            
            # Convert args
            args = []
            for arg in raw_args:
                if arg.lower() == 'true':
                    args.append(True)
                elif arg.lower() == 'false':
                    args.append(False)
                elif arg.lower() == 'none':
                    args.append(None)
                else:
                    try:
                        if '.' in arg:
                            args.append(float(arg))
                        else:
                            args.append(int(arg))
                    except ValueError:
                        args.append(arg) # Keep as string
            
            print(f"Direct Execution: {func_name}({args})")
            self.logger.info(f"Direct Execution: {func_name}({args})")

            # 1. Check Agent Tools (Wrapped functions)
            # self.tools is a list of callables
            tool_func = next((t for t in self.tools if t.__name__ == func_name), None)
            
            if tool_func:
                # Introspect to map args if needed, or just pass *args
                # Since our tools are simple python functions, *args usually works 
                # if the user provided them in order.
                import inspect
                sig = inspect.signature(tool_func)
                
                # Simple binding attempt
                try:
                    bound_args = sig.bind(*args)
                    bound_args.apply_defaults()
                    res = tool_func(*bound_args.args, **bound_args.kwargs)
                    print(f"Result: {res}")
                    self.logger.info(f"Result: {res}")
                    return
                except TypeError as e:
                    print(f"Argument mismatch for tool '{func_name}': {e}")
                    return

            # 2. Check MCP Server directly (underlying functions)
            # This allows calling functions not exposed as agent tools
                if hasattr(self.executor.module, func_name):
                    func = getattr(self.executor.module, func_name)
                    try:
                        res = func(*args)
                        print(f"Result: {res}")
                        self.logger.info(f"Result: {res}")
                        return
                    except Exception as e:
                         print(f"Error executing MCP function '{func_name}': {e}")
                         return

            print(f"Error: Function '{func_name}' not found in Agent Tools or MCP Server.")

        except Exception as e:
            print(f"Failed to execute direct command: {e}")
            traceback.print_exc()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    agent = GeminiAgent()
    print("VR Agent v4 (Multi-Model) Ready.")
    print("Commands: 'white cane' to activate accessibility mode, 'quit' to exit.")
    
    while True:
        try:
            # If keyboard control is active, skip input() — stdin is in cbreak mode.
            # The background thread handles keys. Sleep briefly and loop back.
            if hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl and agent.keyboard_ctrl.active:
                time.sleep(0.1)
                continue

            user_input = input("\nYou: ").strip()
            if not user_input: continue

            cmd = user_input.lower()

            # --- Keyboard VR toggle (backtick) ---
            if cmd == '`' and hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl:
                agent.keyboard_ctrl.activate()  # non-blocking — returns immediately
                continue

            if cmd in ['quit', 'exit']:
                # Deactivate white cane if active
                if agent.white_cane.active:
                    agent.white_cane.deactivate()
                if hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl:
                    agent.keyboard_ctrl.stop()
                break

            elif cmd == 'status':
                agent.print_status()
                continue
            
            # White Cane Activation (Live API + Keyboard)
            elif cmd in ['white cane', 'whitecane', 'enable white cane']:
                print("\n[White Cane] Activating Live API Mode...")
                
                # 1. Activate Keyboard Control
                if hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl:
                    agent.keyboard_ctrl.activate()
                    print("[White Cane] Keyboard control enabled (WASD to move).")
                else:
                    print("[White Cane] Warning: Keyboard control not available.")

                # 2. Start Live API Session (Blocking)
                print("[White Cane] Connecting to Gemini Live... (Say 'stop' or press Ctrl+C to exit)")
                try:
                    asyncio.run(agent.live_white_cane.run())
                except KeyboardInterrupt:
                    pass
                except Exception as e:
                    print(f"[White Cane] Error: {e}")
                
                print("\n[White Cane] Deactivated.")
                continue
            
            # White Cane Deactivation
            elif cmd in ['disable white cane', 'stop white cane', 'exit white cane']:
                result = agent.white_cane.deactivate()
                print(result)
                continue
            
            # White Cane Help (immediate description)
            elif agent.white_cane.active and cmd in ['help', 'what do you see', 'describe', "what's next", 'whats next']:
                print("\nGetting immediate description...")
                description = agent.white_cane.get_immediate_help()
                print(f"\n[White Cane]:\n{description}\n")
                continue
            
            # White Cane Goal Update
            elif agent.white_cane.active and cmd.startswith('goal '):
                new_goal = user_input[5:].strip()
                agent.white_cane.current_goal = new_goal
                print(f"Goal updated: {new_goal}")
                continue
                
            if user_input.startswith("((") and user_input.endswith("))"):
                agent.handle_direct_command(user_input)
                continue

            agent.run(user_input)
        except KeyboardInterrupt:
            # Deactivate white cane if active
            if agent.white_cane.active:
                agent.white_cane.deactivate()
            if hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl:
                agent.keyboard_ctrl.stop()
            break
