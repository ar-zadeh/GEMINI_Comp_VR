
import React from 'react';
import { useSDK } from '../SDKContext';
import { VerificationStatus } from '../types';
import { Activity, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

export const IntentHUD: React.FC = () => {
  const { agentState } = useSDK();

  const getStatusIcon = () => {
    switch (agentState.verificationStatus) {
      case VerificationStatus.SUCCESS: return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case VerificationStatus.FAILED: return <AlertCircle className="w-4 h-4 text-rose-500" />;
      case VerificationStatus.VERIFYING: return <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />;
      default: return <Activity className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="absolute top-1/2 left-8 -translate-y-1/2 w-72 pointer-events-none group">
      <div className="backdrop-blur-xl bg-slate-950/60 border border-white/10 rounded-2xl overflow-hidden shadow-2xl transition-transform duration-500 hover:scale-105">
        {/* Header */}
        <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between bg-white/5">
          <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">Intent_HUD_Core</span>
          {getStatusIcon()}
        </div>

        <div className="p-5 space-y-6">
          {/* Active Action */}
          <section>
            <h4 className="text-[10px] uppercase tracking-widest text-cyan-400 mb-2 font-bold font-mono">Active Action</h4>
            <div className="flex items-center gap-3">
              <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(34,211,238,1)]"></div>
              <p className="text-sm font-medium text-white truncate">{agentState.activeAction}</p>
            </div>
          </section>

          {/* Current Plan */}
          <section>
            <h4 className="text-[10px] uppercase tracking-widest text-slate-400 mb-2 font-bold font-mono">Current Plan</h4>
            <ul className="space-y-3">
              {agentState.plan.map((step, idx) => (
                <li key={idx} className="flex gap-3 items-start group/step">
                  <span className="text-[10px] font-mono text-slate-600 mt-1">{String(idx + 1).padStart(2, '0')}</span>
                  <p className={`text-xs ${idx === 0 ? 'text-white' : 'text-slate-400'} leading-relaxed`}>
                    {step}
                  </p>
                </li>
              ))}
            </ul>
          </section>

          {/* Verification Bar */}
          <section>
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-[10px] uppercase text-slate-500 font-mono font-bold">Verification</span>
              <span className="text-[10px] text-cyan-400 font-mono">{agentState.verificationStatus}</span>
            </div>
            <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-1000 ${agentState.verificationStatus === VerificationStatus.SUCCESS ? 'bg-emerald-400 w-full' :
                    agentState.verificationStatus === VerificationStatus.FAILED ? 'bg-rose-500 w-1/3' :
                      agentState.verificationStatus === VerificationStatus.VERIFYING ? 'bg-cyan-400 w-2/3 animate-[shimmer_2s_infinite]' : 'bg-slate-700 w-0'
                  }`}
              />
            </div>
          </section>
        </div>

        {/* Footer Glitch/Scan Effect */}
        <div className="h-0.5 bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent"></div>
      </div>
    </div>
  );
};
