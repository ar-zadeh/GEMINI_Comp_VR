
import React from 'react';
import { useSDK } from '../SDKContext';
import { Scan, RefreshCw, Terminal, LayoutGrid } from 'lucide-react';

export const CommandWrist: React.FC = () => {
  const { scanScene, rePlan, toggleDebug, isDebugMode } = useSDK();

  const skills = [
    { label: 'Scan Scene', icon: Scan, action: scanScene, color: 'hover:text-cyan-400' },
    { label: 'Force Re-Plan', icon: RefreshCw, action: rePlan, color: 'hover:text-emerald-400' },
    { label: 'Toggle Debug', icon: Terminal, action: toggleDebug, color: isDebugMode ? 'text-cyan-400' : 'hover:text-white' },
    { label: 'Skill Manifest', icon: LayoutGrid, action: () => { }, color: 'hover:text-purple-400' },
  ];

  return (
    <div className="absolute bottom-10 left-10 group pointer-events-auto">
      {/* Wrist Attachment Simulation */}
      <div className="flex items-center gap-1 mb-2 opacity-30 group-hover:opacity-100 transition-opacity">
        <div className="h-[1px] w-12 bg-white"></div>
        <span className="text-[8px] font-bold tracking-[0.3em] uppercase">Left_Wrist_Anchor</span>
      </div>

      <div className="flex flex-col gap-2 p-1.5 backdrop-blur-md bg-white/5 border border-white/10 rounded-xl transform perspective-1000 rotate-y-12 rotate-x-6 hover:rotate-y-0 transition-all duration-500 shadow-xl">
        {skills.map((skill, i) => (
          <button
            key={i}
            onClick={skill.action}
            className={`flex items-center gap-4 px-4 py-3 rounded-lg bg-white/5 border border-transparent hover:border-white/20 transition-all group/btn ${skill.color}`}
          >
            <skill.icon className="w-5 h-5 transition-transform group-hover/btn:scale-110" />
            <span className="text-xs font-medium tracking-wide uppercase text-slate-300 group-hover/btn:text-white">{skill.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
