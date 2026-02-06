
export const COLORS = {
  agentPrimary: '#00f2ff',
  agentGlow: 'rgba(0, 242, 255, 0.4)',
  warning: '#fbbf24',
  danger: '#ef4444',
  success: '#10b981',
  background: '#0f172a',
};

export const ATTENTION_GLOW_SHADER = {
  uniforms: {
    uTime: { value: 0 },
    uColor: { value: [0.0, 0.95, 1.0] },
    uIntensity: { value: 0.5 }
  },
  vertexShader: `
    varying vec2 vUv;
    varying vec3 vNormal;
    void main() {
      vUv = uv;
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float uTime;
    uniform vec3 uColor;
    uniform float uIntensity;
    varying vec2 vUv;
    varying vec3 vNormal;

    void main() {
      float pulse = (sin(uTime * 3.0) + 1.0) * 0.5;
      float edge = 1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0));
      edge = pow(edge, 3.0);
      
      vec3 finalColor = uColor * (0.5 + pulse * 0.5) * edge * 2.0;
      gl_FragColor = vec4(finalColor, edge * uIntensity);
    }
  `
};

export const INITIAL_SKILL_MANIFEST = `
# InteractionKit: Skill Manifest v1.0
- **ScanEnv**: Multi-modal spatial parsing via Gemini-3-Pro.
- **ObjectGrasp**: SAM3-assisted vertex weighting for precise manipulation.
- **AutoCorrect**: Heuristic verification loop with amber visual feedback.
- **SpatialAudio**: Directional cues for out-of-view agent feedback.
`;
