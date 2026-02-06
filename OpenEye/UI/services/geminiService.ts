
import { BoundingBox, AgentState } from '../types';

export class GeminiService {
  private static socket: WebSocket | null = null;
  private static listeners: ((data: any) => void)[] = [];

  static connect(onMessage?: (data: any) => void) {
    if (onMessage) {
      this.listeners.push(onMessage);
    }

    if (this.socket) return;

    this.socket = new WebSocket('ws://localhost:8765');

    this.socket.onopen = () => {
      console.log('[WS] Connected to Gemini VR Backend');
    };

    this.socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.listeners.forEach(l => l(data));
    };

    this.socket.onclose = () => {
      console.log('[WS] Disconnected from Gemini VR Backend');
      this.socket = null;
      // Reconnect after 2 seconds
      setTimeout(() => this.connect(), 2000);
    };
  }

  static addListener(callback: (data: any) => void) {
    this.listeners.push(callback);
  }

  /**
   * Sends a request to the backend agent.
   */
  static sendRequest(request: string) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: 'user_request', content: request }));
    }
  }

  static send(data: any) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    }
  }

  static moveRelative(device: string, dx: number, dy: number, dz: number) {
    this.send({
      type: 'move_relative',
      device,
      dx,
      dy,
      dz
    });
  }

  static triggerAction(device: string, action: string) {
    this.send({
      type: 'trigger_action',
      device,
      action
    });
  }

  static readPose(device?: string) {
    this.send({
      type: 'read_pose',
      device
    });
  }

  static rotateRelative(device: string, dp: number, dy: number, dr: number) {
    this.send({
      type: 'rotate_relative',
      device,
      dp,
      dy,
      dr
    });
  }

  /**
   * Legacy mock - will now be handled by real-time updates.
   */
  static async analyzeScene(): Promise<{ objects: BoundingBox[] }> {
    return { objects: [] };
  }

  static async getRevisedPlan(failureReason: string): Promise<string[]> {
    return [`Correction: ${failureReason}`];
  }
}
