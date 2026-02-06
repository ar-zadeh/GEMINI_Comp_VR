
import React, { useState, useCallback, useMemo, createContext, useContext } from 'react';
import { IntentHUD } from './components/IntentHUD';
import { ManualControl } from './components/ManualControl';
import { CommandWrist } from './components/CommandWrist';
import { FidelityMonitor } from './components/FidelityMonitor';
import { VRCanvas } from './components/VRCanvas';
import { VisionFeed } from './components/VisionFeed';
import { SystemArchitecture } from './components/SystemArchitecture';
import { AgentState, VerificationStatus, ThinkingLevel, MediaResolution, BoundingBox } from './types';
import { GeminiService } from './services/geminiService';
import { SDKContext } from './SDKContext';
import { DebugConsole } from './components/DebugConsole';

const App: React.FC = () => {
  const [agentState, setAgentState] = useState<AgentState>({
    plan: ['Initialize SDK', 'Establish Spatial Anchor', 'Wait for Command'],
    activeAction: 'Standby',
    verificationStatus: VerificationStatus.PENDING,
    thinkingLevel: ThinkingLevel.LOW,
    mediaResolution: MediaResolution.HIGH,
  });

  const [detectedObjects, setDetectedObjects] = useState<BoundingBox[]>([]);
  const [isDebugMode, setIsDebugMode] = useState(false);
  const [showArch, setShowArch] = useState(false);

  const [vrState, setVrState] = useState<any>(null); // Store full VR state
  const [wsMessages, setWsMessages] = useState<any[]>([]);

  // Connect to WebSocket on mount
  React.useEffect(() => {
    GeminiService.connect((data) => {
      // Add to debug logs (reverse chronological preferred for state tracking)
      setWsMessages(current => [
        ...current.slice(-49),
        { ...data, _timestamp: Date.now() }
      ]);

      if (data.type === 'state_update' || data.type === 'pose_data') {
        setVrState(data);
        // Debugging: log consolidated poses
        if (data.poses) {
          console.log('[DEBUG] VR Poses:', {
            headset: data.poses.headset,
            controller1: data.poses.controller1,
            controller2: data.poses.controller2,
            _timestamp: new Date().toLocaleTimeString()
          });
        }
      } else if (data.type === 'agent_state') {
        setAgentState(prev => ({
          ...prev,
          plan: data.plan.length > 0 ? data.plan : prev.plan,
          activeAction: data.activeAction,
          verificationStatus: data.verificationStatus as VerificationStatus,
          thinkingLevel: data.thinkingLevel as ThinkingLevel,
          correctionMessage: data.correctionMessage
        }));
      } else if (data.type === 'vision_update') {
        setAgentState(prev => ({
          ...prev,
          currentFrame: data.frame
        }));
      }
    });
  }, []);

  const scanScene = useCallback(async () => {
    // In the new architecture, we send a command to the agent
    GeminiService.sendRequest("Scan the scene and describe what you see.");
  }, []);

  const rePlan = useCallback(async () => {
    GeminiService.sendRequest("Re-calculate plan based on current situation.");
  }, []);

  const toggleDebug = () => setIsDebugMode(!isDebugMode);

  const contextValue = useMemo(() => ({
    agentState,
    detectedObjects,
    scanScene,
    rePlan,
    toggleDebug,
    isDebugMode,
    setAgentState
  }), [agentState, detectedObjects, scanScene, rePlan, toggleDebug, isDebugMode]);

  return (
    <SDKContext.Provider value={contextValue}>
      <div className="relative w-full h-screen bg-black overflow-hidden select-none">
        {/* Main 3D Simulation */}
        <VRCanvas />

        {/* 2D UI Overlays */}
        <IntentHUD />

        <div className="absolute top-6 left-6 flex flex-col gap-4 pointer-events-none">
          <h1 className="text-xl font-bold tracking-tighter text-cyan-400 font-mono">
            INTERACTION_KIT <span className="text-xs opacity-50 text-white font-normal uppercase tracking-widest px-2 py-0.5 border border-white/20 ml-2">v3.0.0-PRO</span>
          </h1>
          <FidelityMonitor />
        </div>

        <VisionFeed />

        <CommandWrist />
        <ManualControl vrState={vrState} />

        {/* Debug Console overlay */}
        <DebugConsole
          messages={wsMessages}
          onClear={() => setWsMessages([])}
        />

        {/* Self-Correction Flash Overlay */}
        {agentState.correctionMessage && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none animate-pulse">
            <div className="bg-amber-500/10 border-2 border-amber-500/50 backdrop-blur-md px-12 py-8 rounded-2xl">
              <span className="text-amber-500 font-bold text-2xl uppercase tracking-[0.2em] drop-shadow-lg">
                {agentState.correctionMessage}
              </span>
            </div>
          </div>
        )}

        {/* System Arch Toggle */}
        <button
          onClick={() => setShowArch(!showArch)}
          className="absolute bottom-6 right-6 bg-white/5 hover:bg-white/10 border border-white/10 px-4 py-2 rounded text-xs font-mono transition-colors"
        >
          {showArch ? 'HIDE ARCH' : 'SHOW ARCHITECTURE'}
        </button>

        {showArch && <SystemArchitecture onClose={() => setShowArch(false)} />}
      </div>
    </SDKContext.Provider>
  );
};

export default App;
