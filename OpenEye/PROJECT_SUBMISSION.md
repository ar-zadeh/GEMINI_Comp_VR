# OpenEye – AI-Powered VR Accessibility for the Visually Impaired

---

## Inspiration & Problem

Virtual Reality promises immersive experiences for everyone – but what about the millions of people who are blind or visually impaired? Today's VR apps assume you can see the virtual world. There's no accessibility layer, no audio guidance, and no way for a blind person to navigate a social VR space like VRChat independently.

We wanted to change that. **OpenEye** is an AI agent that acts as the "eyes" for a blind VR user, continuously analyzing the virtual environment through the headset camera and providing real-time audio guidance using natural language. Think of it as a guide dog for the metaverse.

The challenge? VR interaction requires precise spatial awareness, quick reactions, and understanding complex 3D environments – far beyond what traditional screen readers or accessibility tools can handle. We needed to build an agent that could *see*, *understand*, *speak*, and *act* – all in real-time.

---

## What It Does

**OpenEye** is an AI-powered VR accessibility system that enables blind and visually impaired users to navigate virtual worlds independently. It combines multiple Gemini models with computer vision, speech recognition, and a custom OpenVR driver to create a complete sensory replacement system.

### 🦯 White Cane Mode (Core Accessibility Feature)
The signature feature – a continuous monitoring mode that acts like a virtual guide dog:
- **Automatic scene capture** every few seconds to build environmental awareness
- **Goal-oriented navigation** – tell it "I want to find the portal" and it guides you there
- **Natural language descriptions** – "There's a door 3 meters ahead, slightly to your left. Two players are standing near the bar to your right."
- **Action recommendations** – "Turn 30 degrees right and walk forward. Watch out for the staircase."
- **Voice-activated commands** – Press a key to ask questions or update your goal

### 🎯 Visual Servoing with SAM 3
When you need to interact with something specific (like a menu button), OpenEye uses:
1. **Gemini Grounding** to locate the target object in the scene
2. **SAM 3 (Segment Anything Model)** for real-time tracking as you move
3. **PID control loop** to automatically align your VR controller ray with the target

This means you can say "click the settings button" and the system will guide your virtual hand to point directly at it.

### 🎤 Full Voice Control
- **Whisper STT** for speech recognition – works offline, fast, and accurate
- **gTTS** for text-to-speech output – every response is spoken aloud
- **Push-to-talk** support for social VR (so you don't broadcast to the game)

### 🕹️ Complete VR Control
OpenEye doesn't just observe – it can take actions:
- Virtual keyboard typing (character-by-character input)
- Controller button presses (trigger, grip, menu, trackpad)
- Joystick movement for locomotion
- Object grabbing and manipulation

---

## How We Built It

### Multi-Model Gemini Architecture
We use **specialized Gemini models for each phase** of the agent loop, each chosen for their strengths:

| Phase | Model | Why |
|-------|-------|-----|
| **Planning** | Gemini 3 Flash Preview | Fast structured output, tool selection |
| **Grounding** | Gemini 3 Flash Preview | Object detection with normalized bounding boxes |
| **Verification** | Gemini 2.5 Flash | High-reasoning validation of actions |
| **Description** | Gemini 2.5 Flash Lite | Fast, natural scene descriptions for TTS |
| **White Cane** | Gemini 3 Flash Preview | Multi-image temporal reasoning for navigation |

### Custom OpenVR Driver (C++)
We built a complete **virtual VR headset and controller driver** that:
- Runs inside SteamVR as a legitimate device driver
- Receives pose commands over TCP from our Python agent
- Captures the VR view and streams it back as base64 JPEG frames
- Handles all button inputs, joystick values, and haptic feedback
- Records audio for voice commands

### MCP Server (Python)
The **Model Context Protocol server** bridges the AI and VR:
- 40+ tools for movement, rotation, button input, vision capture
- Real-time pose broadcasting to the C++ driver
- Vision request/response handling for frame capture
- Thread-safe state management for concurrent operations

### Visual Servoing Pipeline
For precise pointing (needed for UI interaction in VR):

```
1. Gemini 3 Flash → Ground target object + controller ray
2. SAM 3 → Track both objects in real-time video
3. PID Controller → Calculate rotation corrections
4. MCP Server → Apply rotation to virtual controller
5. Repeat until ray overlaps target (within 15px tolerance)
```

### White Cane Architecture
The accessibility mode maintains environmental awareness:

```
1. Background thread captures images every 10 seconds
2. On voice command ("help" / "what's next"):
   - Send last 5 images to Gemini with timestamps
   - Include conversation history for context
   - Structured output: thought, description, action, goal_achieved
3. Speak the description and action recommendation
4. Detect goal changes from user speech
```

---

## Challenges We Overcame

### 1. Real-Time VR Frame Streaming
The OpenVR compositor doesn't expose an easy way to capture what the headset sees. We had to:
- Hook into the texture submission pipeline in the C++ driver
- Compress frames to JPEG in real-time (60fps source → 10fps capture)
- Base64 encode and stream over TCP without blocking the render loop

### 2. Hallucination-Free Object Detection
Early tests with naïve prompting led to Gemini "seeing" objects that weren't there. We solved this by:
- Using **structured output schemas** (Pydantic) to enforce valid responses
- Adding bounding box validation (0-1 normalized coordinates)
- Implementing **SAM 3 tracking** for visual confirmation

### 3. Audio Pipeline Conflicts
Recording user voice while playing TTS was causing feedback loops. We implemented:
- Sequential speech queue to prevent overlapping audio
- Muting the VR microphone during voice commands
- Separate audio streams for input vs output

### 4. Controller Alignment Precision
Pointing a virtual laser at a small UI button requires sub-centimeter accuracy. Our visual servoing loop:
- Uses tip-of-ray detection (highest point in SAM mask)
- Applies proportional control with tuned gains
- Includes divergence detection to abort if things go wrong

---

## What We Learned

### Accessibility Requires Context, Not Just Description
A blind user doesn't need to know *everything* about the scene – they need *actionable* information. "There's a blue cube" is useless. "Step left to avoid the obstacle, then walk forward 3 steps" is helpful. Our White Cane mode focuses on **recommendations**, not just observations.

### Multi-Model Orchestration Works
Using specialized models for each subtask (planning vs. grounding vs. description) produced better results than a single do-everything prompt. The models' strengths complement each other.

### Real VR Accessibility is a Systems Problem
This isn't just an AI demo – it required:
- C++ driver development (OpenVR API)
- Real-time networking (TCP pose streaming)
- Audio engineering (TTS/STT pipeline)
- Computer vision (SAM 3 integration)
- Human factors design (what information is actually useful?)

The hardest part wasn't the AI – it was making all the pieces work together reliably.

---

## What's Next

1. **Live Voice Streaming** – Replace recorded segments with real-time Gemini Live API for conversational interaction
2. **Spatial Audio** – Use 3D sound to indicate object positions ("the door is *here*" with directional audio)
3. **Social VR Integration** – Announce who's speaking, transcribe other players' speech
4. **Hand Tracking** – Support Quest-style hand tracking in addition to controllers
5. **Open Source Release** – Full driver + agent code for the accessibility community

---

## Built With

- **gemini-3-flash-preview** – Planning and object grounding
- **gemini-2.5-flash** – Action verification
- **gemini-2.5-flash-lite-preview** – Scene description
- **sam-3** – Segment Anything Model for object tracking
- **whisper** – OpenAI Whisper for speech-to-text
- **openvr** – SteamVR driver SDK
- **python** – Agent orchestration
- **c++** – Custom VR driver

---

## Try It Out

**Requirements:**
- SteamVR on Windows/Linux
- NVIDIA GPU (for SAM 3)
- Microphone and speakers

```bash
# Clone the repository
git clone https://github.com/your-username/OpenEye.git
cd OpenEye

# Install Python dependencies
pip install -r requirements.txt

# Build and install the OpenVR driver (see driver_sample/README.md)

# Start the agent
python gemini_vr_agent_v8.py

# In the agent, type "white cane" to activate accessibility mode
```

For detailed setup instructions, see the [GitHub Repository](https://github.com/your-username/OpenEye).

---

## The Team

Built with ❤️ for the Gemini API Developer Competition by accessibility advocates who believe the metaverse should be for everyone.
