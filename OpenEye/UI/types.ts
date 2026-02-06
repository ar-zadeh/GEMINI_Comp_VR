
import React from 'react';

export enum VerificationStatus {
  PENDING = 'PENDING',
  VERIFYING = 'VERIFYING',
  SUCCESS = 'SUCCESS',
  FAILED = 'FAILED'
}

export enum ThinkingLevel {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH'
}

export enum MediaResolution {
  LOW = '480p',
  MEDIUM = '720p',
  HIGH = '1080p'
}

export interface AgentState {
  plan: string[];
  activeAction: string;
  verificationStatus: VerificationStatus;
  thinkingLevel: ThinkingLevel;
  mediaResolution: MediaResolution;
  correctionMessage?: string;
  currentFrame?: string;
}

export interface BoundingBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  label?: string;
}

export interface SDKContextType {
  agentState: AgentState;
  detectedObjects: BoundingBox[];
  scanScene: () => Promise<void>;
  rePlan: () => Promise<void>;
  toggleDebug: () => void;
  isDebugMode: boolean;
  /* Added React import above to resolve the missing namespace error on line 45 */
  setAgentState: React.Dispatch<React.SetStateAction<AgentState>>;
}