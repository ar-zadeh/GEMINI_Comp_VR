
import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Billboard, Text } from '@react-three/drei';
import * as THREE from 'three';
import { useSDK } from '../SDKContext';

export const SpatialReticle: React.FC = () => {
  const meshRef = useRef<THREE.Group>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const { detectedObjects, isDebugMode } = useSDK();

  useFrame(({ clock, mouse, camera }) => {
    if (meshRef.current) {
      // Raycasting mock: move reticle with mouse/gaze
      const vector = new THREE.Vector3(mouse.x, mouse.y, 0.5);
      vector.unproject(camera);
      const dir = vector.sub(camera.position).normalize();
      const distance = 5;
      const pos = camera.position.clone().add(dir.multiplyScalar(distance));
      meshRef.current.position.copy(pos);
    }

    if (ringRef.current) {
      ringRef.current.rotation.z = clock.getElapsedTime();
    }
  });

  return (
    <group ref={meshRef}>
      <Billboard follow={true}>
        {/* Outer Ring */}
        <mesh ref={ringRef}>
          <ringGeometry args={[0.2, 0.22, 32]} />
          <meshBasicMaterial color="#00f2ff" transparent opacity={0.6} side={THREE.DoubleSide} />
        </mesh>

        {/* Inner Crosshair */}
        <mesh>
          <ringGeometry args={[0, 0.02, 32]} />
          <meshBasicMaterial color="#00f2ff" />
        </mesh>

        {/* Dynamic Label */}
        <Text
          position={[0, -0.4, 0]}
          fontSize={0.08}
          color="#00f2ff"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v13/t6X24m62m6y8-o71b87y9059_e78.woff"
          anchorX="center"
          anchorY="middle"
        >
          {detectedObjects.length > 0 ? 'TARGET_LOCKED' : 'IDLE_SCAN'}
        </Text>

        {/* Bounding Box Visualizer for Debug */}
        {isDebugMode && detectedObjects.map((obj, i) => (
          <mesh key={i} position={[0, 0, -0.1]}>
            <planeGeometry args={[1, 1]} />
            <meshBasicMaterial color="cyan" wireframe opacity={0.2} transparent />
          </mesh>
        ))}
      </Billboard>
    </group>
  );
};
