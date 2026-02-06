import React, { useState, useEffect, useRef } from 'react';
import { Terminal, ChevronDown, ChevronUp, Trash2, X } from 'lucide-react';

interface DebugConsoleProps {
    messages: any[];
    onClear: () => void;
}

export const DebugConsole: React.FC<DebugConsoleProps> = ({ messages, onClear }) => {
    const [isOpen, setIsOpen] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (scrollRef.current && isOpen) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isOpen]);

    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                className="fixed bottom-6 left-6 p-3 rounded-full bg-slate-900/80 border border-white/10 text-cyan-400 hover:bg-cyan-400/10 transition-all shadow-2xl backdrop-blur-md group"
                title="Open Debug Console"
            >
                <Terminal className="w-5 h-5 group-hover:scale-110 transition-transform" />
                {messages.length > 0 && (
                    <span className="absolute -top-1 -right-1 w-4 h-4 bg-cyan-500 text-white text-[10px] flex items-center justify-center rounded-full font-bold">
                        {messages.length > 99 ? '99+' : messages.length}
                    </span>
                )}
            </button>
        );
    }

    return (
        <div className="fixed bottom-6 left-6 w-96 h-80 flex flex-col bg-slate-950/90 border border-white/10 rounded-2xl shadow-2xl backdrop-blur-xl overflow-hidden z-50">
            {/* Header */}
            <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between bg-white/5">
                <div className="flex items-center gap-2">
                    <Terminal className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">WS_CONSOLE</span>
                </div>
                <div className="flex items-center gap-1">
                    <button onClick={onClear} className="p-1 hover:bg-white/5 rounded transition-colors text-slate-500 hover:text-red-400" title="Clear Logs">
                        <Trash2 className="w-3.5 h-3.5" />
                    </button>
                    <button onClick={() => setIsOpen(false)} className="p-1 hover:bg-white/5 rounded transition-colors text-slate-500 hover:text-white">
                        <X className="w-3.5 h-3.5" />
                    </button>
                </div>
            </div>

            {/* Message List */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-3 space-y-2 font-mono text-[10px] scrollbar-thin scrollbar-thumb-white/10"
            >
                {messages.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-slate-600 italic">
                        Waiting for messages...
                    </div>
                ) : (
                    messages.map((msg, i) => (
                        <div key={i} className="border-l-2 border-cyan-500/30 pl-2 py-0.5 group">
                            <div className="flex justify-between items-start">
                                <span className={`font-bold ${msg.type === 'pose_data' ? 'text-purple-400' :
                                    msg.type === 'state_update' ? 'text-blue-400' :
                                        'text-emerald-400'
                                    }`}>
                                    {msg.type}
                                </span>
                                <span className="text-slate-600 text-[8px]">
                                    {new Date(msg._timestamp || Date.now()).toLocaleTimeString()}
                                </span>
                            </div>
                            <pre className="text-slate-400 mt-0.5 whitespace-pre-wrap break-all max-h-24 overflow-y-auto">
                                {JSON.stringify(msg, (key, value) => {
                                    if (key === 'frame' && typeof value === 'string' && value.length > 100) {
                                        return '{...}'; // Still truncate large vision frames
                                    }
                                    return value;
                                }, 2)}
                            </pre>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
