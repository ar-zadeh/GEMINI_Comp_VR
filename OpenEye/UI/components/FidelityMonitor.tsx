
import React from 'react';
import { useSDK } from '../SDKContext';
import { ThinkingLevel } from '../types';
import { Cpu, Wifi } from 'lucide-react';

export const FidelityMonitor: React.FC = () => {
  const { agentState } = useSDK();

  const getThinkingColor = (level: ThinkingLevel) => {
    switch (level) {
      case ThinkingLevel.HIGH: return 'text-purple-400';
      case ThinkingLevel.MEDIUM: return 'text-cyan-400';
      case ThinkingLevel.LOW: return 'text-slate-400';
      default: return 'text-slate-500';
    }
  };

  return (
    <div className="flex gap-4">
      <div className="flex items-center gap-3 px-3 py-2 bg-slate-900/50 border border-white/5 rounded-lg backdrop-blur-md">
        <Wifi className="w-4 h-4 text-slate-500" />
        <div className="flex flex-col">
          <span className="text-[8px] uppercase font-bold text-slate-500 tracking-widest">Media_Res</span>
          <span className="text-[10px] font-mono text-white">{agentState.mediaResolution}</span>
        </div>
      </div>

      <div className="flex items-center gap-3 px-3 py-2 bg-slate-900/50 border border-white/5 rounded-lg backdrop-blur-md">
        <Cpu className={`w-4 h-4 ${getThinkingColor(agentState.thinkingLevel)}`} />
        <div className="flex flex-col">
          <span className="text-[8px] uppercase font-bold text-slate-500 tracking-widest">Thinking_Lvl</span>
          <span className={`text-[10px] font-mono ${getThinkingColor(agentState.thinkingLevel)}`}>
            {agentState.thinkingLevel}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3 px-3 py-2 bg-slate-900/50 border border-white/5 rounded-lg backdrop-blur-md">
        <div className="flex flex-col items-end">
          <span className="text-[8px] uppercase font-bold text-slate-500 tracking-widest">Latency</span>
          <span className="text-[10px] font-mono text-emerald-400">12ms</span>
        </div>
      </div>
    </div>
  );
};
