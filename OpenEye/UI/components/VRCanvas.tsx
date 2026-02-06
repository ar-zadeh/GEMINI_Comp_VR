
import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Float, MeshDistortMaterial, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';
import { useSDK } from '../SDKContext';
import { SpatialReticle } from './SpatialReticle';
import { ATTENTION_GLOW_SHADER } from '../constants';

const AgentAttentionObject: React.FC<{ position: [number, number, number], color: string, id: string }> = ({ position, color, id }) => {
  const mesh = useRef<THREE.Mesh>(null);
  const shaderRef = useRef<THREE.ShaderMaterial>(null);
  const { isDebugMode, detectedObjects } = useSDK();

  const isDetected = useMemo(() => {
    // Mocking detection match for visual feedback
    return detectedObjects.length > 0 && Math.random() > 0.5;
  }, [detectedObjects]);

  useFrame((state) => {
    if (mesh.current) {
      mesh.current.rotation.x += 0.005;
      mesh.current.rotation.y += 0.005;
    }
    if (shaderRef.current) {
      shaderRef.current.uniforms.uTime.value = state.clock.getElapsedTime();
    }
  });

  return (
    <group position={position}>
      <mesh ref={mesh}>
        <octahedronGeometry args={[0.5, 0]} />
        <MeshDistortMaterial color={color} speed={2} distort={0.2} radius={1} />
      </mesh>

      {/* Agent Attention Glow Shader Layer */}
      {isDetected && (
        <mesh scale={[1.3, 1.3, 1.3]}>
          <octahedronGeometry args={[0.5, 0]} />
          <shaderMaterial
            ref={shaderRef}
            transparent
            depthWrite={false}
            vertexShader={ATTENTION_GLOW_SHADER.vertexShader}
            fragmentShader={ATTENTION_GLOW_SHADER.fragmentShader}
            uniforms={ATTENTION_GLOW_SHADER.uniforms}
          />
        </mesh>
      )}

      {isDebugMode && (
        <gridHelper args={[2, 10, 0x00f2ff, 0x1e293b]} rotation={[Math.PI / 2, 0, 0]} />
      )}
    </group>
  );
};

export const VRCanvas: React.FC = () => {
  return (
    <div className="w-full h-full cursor-crosshair">
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[0, 0, 8]} fov={50} />
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={3}
          maxDistance={15}
          autoRotate={false}
          maxPolarAngle={Math.PI / 1.5}
          minPolarAngle={Math.PI / 4}
        />

        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} intensity={1.5} castShadow />
        <spotLight position={[-10, 10, 5]} angle={0.15} penumbra={1} intensity={1} castShadow />

        <Float speed={1.5} rotationIntensity={0.5} floatIntensity={0.5}>
          <AgentAttentionObject position={[-2, 1, 0]} color="#00f2ff" id="obj-1" />
          <AgentAttentionObject position={[2, -1, 0]} color="#8b5cf6" id="obj-2" />
          <AgentAttentionObject position={[0, 2, -2]} color="#ec4899" id="obj-3" />
        </Float>

        <ContactShadows position={[0, -2.5, 0]} opacity={0.4} scale={20} blur={2} far={4.5} />

        <Environment preset="night" />

        <SpatialReticle />
      </Canvas>
    </div>
  );
};
