
import React, { useState, useEffect, useCallback } from 'react';
import { GeminiService } from '../services/geminiService';
import { Move, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Target, Gamepad2, MousePointer2 } from 'lucide-react';

interface ManualControlProps {
    vrState: any;
}

export const ManualControl: React.FC<ManualControlProps> = ({ vrState }) => {
    const [selectedDevice, setSelectedDevice] = useState<'headset' | 'controller1' | 'controller2'>('controller1');
    const [stepSize, setStepSize] = useState(0.05);

    useEffect(() => {
        if (vrState?.type === 'pose_data') {
            console.log('[ManualControl] Explicit Pose Read Result:', vrState.poses);
        }
    }, [vrState]);

    const move = useCallback((dx: number, dy: number, dz: number) => {
        GeminiService.moveRelative(selectedDevice, dx, dy, dz);
    }, [selectedDevice]);

    const rotate = useCallback((dp: number, dy: number, dr: number) => {
        GeminiService.rotateRelative(selectedDevice, dp, dy, dr);
    }, [selectedDevice]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Ignore if typing in an input
            if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') return;

            const rotateStep = 5; // 5 degrees

            switch (e.key) {
                // Movement (WASDQE)
                case 'w': case 'W': move(0, 0, -stepSize); break;
                case 's': case 'S': move(0, 0, stepSize); break;
                case 'a': case 'A': move(-stepSize, 0, 0); break;
                case 'd': case 'D': move(stepSize, 0, 0); break;
                case 'q': case 'Q': move(0, -stepSize, 0); break;
                case 'e': case 'E': move(0, stepSize, 0); break;

                // Rotation (Arrows)
                case 'ArrowUp': rotate(rotateStep, 0, 0); e.preventDefault(); break;
                case 'ArrowDown': rotate(-rotateStep, 0, 0); e.preventDefault(); break;
                case 'ArrowLeft': rotate(0, -rotateStep, 0); e.preventDefault(); break;
                case 'ArrowRight': rotate(0, rotateStep, 0); e.preventDefault(); break;

                // Device Selection
                case '1': setSelectedDevice('headset'); e.preventDefault(); break;
                case '2': setSelectedDevice('controller1'); e.preventDefault(); break;
                case '3': setSelectedDevice('controller2'); e.preventDefault(); break;

                // Action
                case ' ': // Space for trigger
                    GeminiService.triggerAction(selectedDevice, 'click');
                    e.preventDefault();
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [move, stepSize, selectedDevice]);

    return (
        <div className="absolute top-6 right-6 w-64 pointer-events-auto select-none">
            <div className="backdrop-blur-xl bg-slate-950/60 border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
                {/* Header */}
                <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between bg-white/5">
                    <div className="flex items-center gap-2">
                        <Move className="w-4 h-4 text-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.5)]" />
                        <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">Manual_Control</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <span className="text-[8px] font-mono text-emerald-400/70 uppercase">Linked</span>
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
                    </div>
                </div>

                <div className="p-4 space-y-5">
                    {/* Device Selector */}
                    <div className="space-y-2">
                        <span className="text-[8px] uppercase font-bold text-slate-500 tracking-[0.2em] ml-1">Active Device</span>
                        <div className="grid grid-cols-3 gap-1 p-1 bg-slate-900/50 rounded-lg border border-white/5">
                            {[
                                { id: 'headset', label: 'HMD', key: '1', icon: Target },
                                { id: 'controller1', label: 'LEFT', key: '2', icon: Gamepad2 },
                                { id: 'controller2', label: 'RIGHT', key: '3', icon: Gamepad2 }
                            ].map(dev => (
                                <button
                                    key={dev.id}
                                    onClick={() => setSelectedDevice(dev.id as any)}
                                    className={`flex flex-col items-center gap-1 py-2 rounded-md transition-all border ${selectedDevice === dev.id
                                        ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 shadow-[inset_0_0_10px_rgba(34,211,238,0.05)]'
                                        : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5'
                                        }`}
                                >
                                    <dev.icon className={`w-3.5 h-3.5 ${selectedDevice === dev.id ? 'text-cyan-400' : 'text-slate-600'}`} />
                                    <span className="text-[9px] font-bold">{dev.label}</span>
                                    <span className="text-[7px] opacity-40">[{dev.key}]</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* D-Pad Simulation */}
                    <div className="space-y-3">
                        <div className="flex justify-between items-center px-1">
                            <span className="text-[8px] uppercase font-bold text-slate-500 tracking-[0.2em]">Spatial Navigation</span>
                            <span className="text-[8px] font-mono text-cyan-400/50">WASDQE</span>
                        </div>

                        <div className="flex flex-col items-center gap-1.5">
                            {/* Vertical Up */}
                            <button
                                onClick={() => move(0, stepSize, 0)}
                                className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-900/60 border border-white/10 hover:border-cyan-500/50 hover:bg-cyan-500/5 active:scale-95 transition-all text-slate-400 hover:text-cyan-400 group"
                                title="Move Up (E)"
                            >
                                <ChevronUp className="w-5 h-5 group-hover:drop-shadow-[0_0_5px_rgba(34,211,238,0.5)]" />
                            </button>

                            {/* Forward */}
                            <button
                                onClick={() => move(0, 0, -stepSize)}
                                className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-900/60 border border-white/10 hover:border-cyan-500/50 hover:bg-cyan-500/5 active:scale-95 transition-all text-slate-400 hover:text-cyan-400 group mt-1"
                                title="Move Forward (W)"
                            >
                                <div className="relative">
                                    <ChevronUp className="w-5 h-5 group-hover:drop-shadow-[0_0_5px_rgba(34,211,238,0.5)]" />
                                    <span className="absolute -top-1 -right-1 w-1.5 h-1.5 bg-cyan-400/30 rounded-full"></span>
                                </div>
                            </button>

                            {/* Horizontal row */}
                            <div className="flex gap-1.5">
                                <button
                                    onClick={() => move(-stepSize, 0, 0)}
                                    className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-900/60 border border-white/10 hover:border-cyan-500/50 hover:bg-cyan-500/5 active:scale-95 transition-all text-slate-400 hover:text-cyan-400 group"
                                    title="Move Left (A)"
                                >
                                    <ChevronLeft className="w-5 h-5 group-hover:drop-shadow-[0_0_5px_rgba(34,211,238,0.5)]" />
                                </button>
                                <button
                                    onClick={() => move(0, 0, stepSize)}
                                    className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-900/60 border border-white/10 hover:border-cyan-500/50 hover:bg-cyan-500/5 active:scale-95 transition-all text-slate-400 hover:text-cyan-400 group"
                                    title="Move Backward (S)"
                                >
                                    <ChevronDown className="w-5 h-5 group-hover:drop-shadow-[0_0_5px_rgba(34,211,238,0.5)]" />
                                </button>
                                <button
                                    onClick={() => move(stepSize, 0, 0)}
                                    className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-900/60 border border-white/10 hover:border-cyan-500/50 hover:bg-cyan-500/5 active:scale-95 transition-all text-slate-400 hover:text-cyan-400 group"
                                    title="Move Right (D)"
                                >
                                    <ChevronRight className="w-5 h-5 group-hover:drop-shadow-[0_0_5px_rgba(34,211,238,0.5)]" />
                                </button>
                            </div>

                            {/* Vertical Down */}
                            <button
                                onClick={() => move(0, -stepSize, 0)}
                                className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-900/60 border border-white/10 hover:border-cyan-500/50 hover:bg-cyan-500/5 active:scale-95 transition-all text-slate-400 hover:text-cyan-400 group mt-1"
                                title="Move Down (Q)"
                            >
                                <ChevronDown className="w-5 h-5 group-hover:drop-shadow-[0_0_5px_rgba(34,211,238,0.5)]" />
                            </button>
                        </div>
                    </div>

                    {/* Trigger Action */}
                    <div className="pt-2">
                        <button
                            onClick={() => GeminiService.triggerAction(selectedDevice, 'click')}
                            className="w-full py-3.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 hover:bg-cyan-500/20 hover:border-cyan-500/50 text-cyan-400 font-bold text-[10px] tracking-[0.2em] uppercase transition-all flex items-center justify-center gap-3 group relative overflow-hidden"
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:animate-[shimmer_2s_infinite]"></div>
                            <MousePointer2 className="w-4 h-4 group-hover:scale-110 transition-transform" />
                            Trigger Action
                        </button>
                    </div>

                    {/* Position Reader */}
                    <div className="space-y-3 pt-2">
                        <div className="flex justify-between items-center px-1">
                            <span className="text-[8px] uppercase font-bold text-slate-500 tracking-[0.2em]">Position Reader</span>
                            <button
                                onClick={() => GeminiService.readPose()}
                                className="text-[8px] font-mono text-cyan-400/70 hover:text-cyan-400 transition-colors uppercase border border-cyan-400/20 px-1.5 py-0.5 rounded"
                            >
                                READ_POSE
                            </button>
                        </div>

                        <div className="space-y-2.5 bg-slate-900/50 p-2.5 rounded-xl border border-white/5">
                            {['headset', 'controller1', 'controller2'].map(dev => {
                                const pose = vrState?.poses?.[dev];
                                const label = dev === 'headset' ? 'HMD' : dev === 'controller1' ? 'L_CTRL' : 'R_CTRL';
                                return (
                                    <div key={dev} className="flex flex-col gap-1 border-b border-white/5 last:border-0 pb-1.5 last:pb-0">
                                        <div className="flex justify-between items-center">
                                            <span className={`text-[7px] font-bold ${selectedDevice === dev ? 'text-cyan-400' : 'text-slate-500'}`}>{label}</span>
                                            <span className="text-[6px] font-mono text-slate-600">{pose ? 'ONLINE' : 'OFFLINE'}</span>
                                        </div>
                                        <div className="grid grid-cols-2 gap-x-2">
                                            <div className="flex justify-between items-baseline bg-black/20 px-1.5 py-0.5 rounded">
                                                <span className="text-[5px] text-slate-600 uppercase font-bold">POS</span>
                                                <span className="text-[8px] font-mono text-slate-300">
                                                    {pose?.pos ? `${pose.pos[0].toFixed(2)} ${pose.pos[1].toFixed(2)} ${pose.pos[2].toFixed(2)}` : '0.00 0.00 0.00'}
                                                </span>
                                            </div>
                                            <div className="flex justify-between items-baseline bg-black/20 px-1.5 py-0.5 rounded">
                                                <span className="text-[5px] text-slate-600 uppercase font-bold">ROT</span>
                                                <span className="text-[8px] font-mono text-slate-300">
                                                    {pose?.rot ? `${pose.rot[0].toFixed(0)}° ${pose.rot[1].toFixed(0)}° ${pose.rot[2].toFixed(0)}°` : '0° 0° 0°'}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Footer Info */}
                    <div className="pt-3 border-t border-white/5 flex justify-between items-center px-1">
                        <div className="flex flex-col">
                            <span className="text-[7px] uppercase font-bold text-slate-600 tracking-widest">Resolution</span>
                            <span className="text-[9px] font-mono text-slate-400">Step: {(stepSize * 100).toFixed(0)}cm</span>
                        </div>
                        <div className="flex gap-1">
                            <button
                                onClick={() => setStepSize(prev => Math.max(0.01, prev - 0.01))}
                                className="w-6 h-6 flex items-center justify-center rounded bg-white/5 border border-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
                            >-</button>
                            <button
                                onClick={() => setStepSize(prev => Math.min(0.5, prev + 0.01))}
                                className="w-6 h-6 flex items-center justify-center rounded bg-white/5 border border-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
                            >+</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
