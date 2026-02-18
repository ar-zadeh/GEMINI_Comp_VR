# ============================================================================
# LIVE API WHITE CANE AGENT
# ============================================================================

class LiveWhiteCaneAgent:
    """
    Handles real-time audio/video interaction using Gemini Live API.
    Replaces the static 'WhiteCaneAssistant' when active.
    """
    def __init__(self, client, executor, log_dir: Path):
        self.client = client
        self.executor = executor
        self.log_dir = log_dir / "live_white_cane"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.model = "gemini-2.5-flash-native-audio-preview-12-2025"
        self.system_instruction = """
        You are a helpful and friendly AI assistant for a blind user in VR. 
        You receive a video feed of what the user is facing in the virtual world.
        Your goal is to help them navigate.
        
        When the user asks "How do I get to X?", look at the video and guide them.
        When the user asks "Where am I?", describe the surroundings.
        
        Be concise. Directions should be clear (e.g., "Turn left", "Walk forward").
        """
        
        # Audio Configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.SEND_SAMPLE_RATE = 16000
        self.RECEIVE_SAMPLE_RATE = 24000
        self.CHUNK_SIZE = 1024
        
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None
        self.active = False

    async def listen_audio(self):
        """Microphone input loop."""
        pya = pyaudio.PyAudio()
        mic_info = pya.get_default_input_device_info()
        stream = await asyncio.to_thread(
            pya.open,
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.CHUNK_SIZE,
        )
        
        try:
            while self.active:
                data = await asyncio.to_thread(stream.read, self.CHUNK_SIZE, exception_on_overflow=False)
                payload = {"data": data, "mime_type": "audio/pcm"}
                try:
                    self.out_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    # Drop oldest audio if queue full to maintain real-time
                    _ = self.out_queue.get_nowait()
                    self.out_queue.put_nowait(payload)
                    
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            pya.terminate()

    async def play_audio(self):
        """Speaker output loop."""
        pya = pyaudio.PyAudio()
        stream = await asyncio.to_thread(
            pya.open,
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RECEIVE_SAMPLE_RATE,
            output=True,
        )
        try:
            while self.active:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            pya.terminate()

    async def capture_vr_view(self):
        """Capture VR view loop (simulated video stream)."""
        logger = get_logger()
        while self.active:
            try:
                # 1. Inspect Surroundings (capture image)
                # This returns a JSON string with base64 encoded image
                res_str = await asyncio.to_thread(self.executor.call, "inspect_surroundings")
                
                try:
                    res = json.loads(res_str)
                    data_b64 = res.get("data")
                    if data_b64:
                         # Send as JPEG
                        await self.out_queue.put({
                            "mime_type": "image/jpeg", 
                            "data": data_b64
                        })
                except json.JSONDecodeError:
                    pass

                # Limit to ~1 FPS to save bandwidth/tokens and match Live API capabilities
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Live Video Capture Error: {e}")
                await asyncio.sleep(1.0)

    async def send_realtime(self):
        """Send audio/video from out_queue to Gemini."""
        try:
            while self.active:
                msg = await self.out_queue.get()
                if msg["mime_type"].startswith("audio/"):
                    await self.session.send_realtime_input(audio=msg)
                else:
                    await self.session.send_realtime_input(media=msg)
        except asyncio.CancelledError:
            pass

    async def receive_loop(self):
        """Receive audio/text from Gemini."""
        try:
            while self.active:
                async for response in self.session.receive():
                    if response.server_content:
                        model_turn = response.server_content.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                if part.executable_code or part.code_execution_result:
                                     # Not implementing code execution for now
                                     continue
                                if part.text:
                                    print(part.text, end="", flush=True)

                    if response.tool_call:
                         # Not implementing tool calls yet, but this is where they'd go
                         print(f"\n[Tool Call]: {response.tool_call}")

                    # Audio data comes in `data` field of the response struct, 
                    # but depending on SDK version it might be in different places.
                    # The example used `if data := response.data:`
                    if response.data:
                        self.audio_in_queue.put_nowait(response.data)

        except asyncio.CancelledError:
            pass

    async def run(self):
        """Main async entry point."""
        self.active = True
        
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=self.system_instruction)]),
        )

        try:
            async with (
                self.client.aio.live.connect(model=self.model, config=config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                print("\n[Live API] Connected! You can speak now.")
                
                tg.create_task(self.listen_audio()) # Mic -> Gemini
                tg.create_task(self.play_audio())   # Gemini -> Speaker
                tg.create_task(self.capture_vr_view()) # VR -> Gemini
                tg.create_task(self.send_realtime()) # Queue -> Session
                tg.create_task(self.receive_loop())  # Session -> Queue/Print
                
                # Keep running until active is set to False (by external stop)
                while self.active:
                    await asyncio.sleep(0.1)
                
                raise asyncio.CancelledError("Stopped by user")

        except asyncio.CancelledError:
            print("\n[Live API] Stopping...")
        except Exception as e:
            print(f"\n[Live API] Error: {e}")
            traceback.print_exc()
        finally:
            self.active = False
