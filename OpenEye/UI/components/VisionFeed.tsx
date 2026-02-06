
import React from 'react';
import { useSDK } from '../SDKContext';

export const VisionFeed: React.FC = () => {
  const { agentState } = useSDK();
  const { currentFrame } = agentState;

  if (!currentFrame) {
    return (
      <div className="vision-feed-loading">
        <div className="loading-text">
          Waiting for Vision Signal...
        </div>
        <div className="overlay-border"></div>
        <style dangerouslySetInnerHTML={{
          __html: `
          .vision-feed-loading {
            position: absolute;
            top: 24px;
            right: 24px;
            width: 256px;
            height: 192px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(0, 242, 255, 0.3);
            backdrop-filter: blur(8px);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
          }
          .loading-text {
            font-size: 10px;
            font-family: monospace;
            color: rgba(0, 242, 255, 0.5);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            animation: pulse-simple 2s infinite;
          }
          .overlay-border {
            position: absolute;
            inset: 0;
            border: 1px solid rgba(0, 242, 255, 0.1);
            pointer-events: none;
          }
          @keyframes pulse-simple {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 0.2; }
          }
        `}} />
      </div>
    );
  }

  return (
    <div className="vision-feed-container">
      {/* Feed Metadata */}
      <div className="feed-header">
        <span className="feed-title">
          Agent_Eye_V.2.1
        </span>
        <div className="live-indicator-container">
          <div className="live-dot"></div>
          <span className="live-text">Live</span>
        </div>
      </div>

      {/* Frame Image */}
      <img
        src={`data:image/jpeg;base64,${currentFrame}`}
        alt="Agent Vision"
        className="feed-image"
      />

      {/* Futuristic Overlays */}
      <div className="feed-overlay">
        {/* Scanning Line */}
        <div className="scan-line"></div>

        {/* Corner Brackets */}
        <div className="bracket top-left"></div>
        <div className="bracket top-right"></div>
        <div className="bracket bottom-left"></div>
        <div className="bracket bottom-right"></div>

        {/* HUD Elements */}
        <div className="hud-info">
          <div className="hud-stat">RES: 512x512</div>
          <div className="hud-stat">LATENCY: 52ms</div>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{
        __html: `
        .vision-feed-container {
          position: absolute;
          top: 24px;
          right: 24px;
          width: 256px;
          height: 192px;
          background: rgba(0, 0, 0, 0.6);
          border: 1px solid rgba(0, 242, 255, 0.5);
          backdrop-filter: blur(8px);
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 0 20px rgba(0, 242, 255, 0.15);
          z-index: 50;
        }
        .feed-header {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          padding: 8px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          z-index: 10;
          background: linear-gradient(to bottom, rgba(0,0,0,0.8), transparent);
        }
        .feed-title {
          font-size: 9px;
          font-family: monospace;
          color: #00f2ff;
          font-weight: bold;
          text-transform: uppercase;
        }
        .live-indicator-container {
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .live-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #ef4444;
          animation: pulse-red 1.5s infinite;
          box-shadow: 0 0 5px rgba(239, 68, 68, 0.5);
        }
        .live-text {
          font-size: 8px;
          font-family: monospace;
          color: rgba(255, 255, 255, 0.7);
          text-transform: uppercase;
        }
        .feed-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: transform 0.7s;
        }
        .vision-feed-container:hover .feed-image {
          transform: scale(1.05);
        }
        .feed-overlay {
          position: absolute;
          inset: 0;
          pointer-events: none;
        }
        .scan-line {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 1px;
          background: rgba(0, 242, 255, 0.3);
          box-shadow: 0 0 5px rgba(0, 242, 255, 0.5);
          animation: scan-move 4s linear infinite;
        }
        .bracket {
          position: absolute;
          width: 12px;
          height: 12px;
          border-color: rgba(0, 242, 255, 0.5);
          border-style: solid;
          border-width: 0;
        }
        .top-left { top: 8px; left: 8px; border-top-width: 1px; border-left-width: 1px; }
        .top-right { top: 8px; right: 8px; border-top-width: 1px; border-right-width: 1px; }
        .bottom-left { bottom: 8px; left: 8px; border-bottom-width: 1px; border-left-width: 1px; }
        .bottom-right { bottom: 8px; right: 8px; border-bottom-width: 1px; border-right-width: 1px; }
        
        .hud-info {
          position: absolute;
          bottom: 8px;
          left: 8px;
          display: flex;
          flex-direction: column;
          gap: 2px;
          opacity: 0.7;
        }
        .hud-stat {
          font-size: 7px;
          font-family: monospace;
          color: #00f2ff;
        }
        @keyframes scan-move {
          0% { top: 0; }
          100% { top: 100%; }
        }
        @keyframes pulse-red {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.2); }
        }
      `}} />
    </div>
  );
};
