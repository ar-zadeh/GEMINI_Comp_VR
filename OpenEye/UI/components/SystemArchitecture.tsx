
import React from 'react';
import { X, Cpu, Layers, Eye, Share2 } from 'lucide-react';
import { INITIAL_SKILL_MANIFEST } from '../constants';

export const SystemArchitecture: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-black/80 backdrop-blur-md overflow-y-auto">
      <div className="max-w-4xl w-full bg-slate-900 border border-white/10 rounded-3xl overflow-hidden shadow-2xl relative">
        <button 
          onClick={onClose}
          className="absolute top-6 right-6 p-2 hover:bg-white/10 rounded-full transition-colors"
        >
          <X className="w-6 h-6" />
        </button>

        <div className="p-12">
          <h2 className="text-3xl font-bold tracking-tight mb-8 flex items-center gap-4">
             <Layers className="text-cyan-400" /> System Architecture
          </h2>

          <div className="grid md:grid-cols-2 gap-12">
            <div className="space-y-8">
              <div className="space-y-4">
                <h3 className="text-lg font-bold text-cyan-400 font-mono flex items-center gap-2">
                  <Cpu className="w-5 h-5" /> REASONING_CORE
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  Powered by <strong>Gemini-3-Pro</strong> for long-horizon planning and <strong>Gemini-3-Flash</strong> for low-latency visual-to-action mapping. InteractionKit uses a dual-loop control system ensuring agent intentions are verified before execution.
                </p>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-bold text-purple-400 font-mono flex items-center gap-2">
                  <Eye className="w-5 h-5" /> PERCEPTION_ENGINE
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  Integrates <strong>SAM3</strong> (Segment Anything Model) for pixel-perfect object isolation in 3D space. Gaze-tracking coordinates are projected into the agent's viewport to focus LLM attention.
                </p>
              </div>

              <div className="p-6 bg-slate-950/50 rounded-xl border border-white/5 space-y-4">
                 <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500">Auto-Correction Logic</h4>
                 <div className="flex gap-4 items-center">
                    <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></div>
                    {/* Fixed "Cannot find name 'FLASH_WARNING'" and 'Amber' by wrapping the pseudo-code string in quotes to stop JSX interpretation of the curly braces */}
                    <p className="text-xs text-slate-300 font-mono">{"IF (Action_Result != Intended_State) { FLASH_WARNING(Amber); RE_PLAN(); }"}</p>
                 </div>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-bold text-emerald-400 font-mono flex items-center gap-2">
                <Share2 className="w-5 h-5" /> SKILL_MANIFEST
              </h3>
              <div className="p-8 bg-slate-950 rounded-2xl border border-white/5 font-mono text-sm overflow-hidden relative">
                <div className="absolute top-0 right-0 p-4 opacity-5 text-emerald-400">
                  <Terminal className="w-24 h-24" />
                </div>
                <pre className="text-slate-300 whitespace-pre-wrap leading-relaxed">
                  {INITIAL_SKILL_MANIFEST}
                </pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const Terminal = (props: any) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="4 17 10 11 4 5"></polyline>
    <line x1="12" y1="19" x2="20" y2="19"></line>
  </svg>
);