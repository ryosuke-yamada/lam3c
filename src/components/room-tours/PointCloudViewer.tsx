import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Button } from '@/components/ui/button';

interface PointCloudViewerProps {
  plyPath: string;
}

export default function PointCloudViewer({ plyPath }: PointCloudViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    setIsLoading(true);
    setError(null);

    // Setup scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // Setup camera
    const camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / container.clientHeight,
      1,
      2000
    );
    camera.position.set(0, 0, 100);
    cameraRef.current = camera;

    // Setup renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Setup controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Load PLY file
    const loader = new PLYLoader();
    let geometry: THREE.BufferGeometry | null = null;
    let material: THREE.ShaderMaterial | null = null;
    let points: THREE.Points | null = null;

    loader.load(
      plyPath,
      (loadedGeometry) => {
        geometry = loadedGeometry;

        // Center the geometry
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox;
        if (bbox) {
          const center = new THREE.Vector3();
          bbox.getCenter(center);
          geometry.translate(-center.x, -center.y, -center.z);

          // Calculate proper camera distance based on bounding box
          const size = new THREE.Vector3();
          bbox.getSize(size);
          const maxDim = Math.max(size.x, size.y, size.z);
          const fov = camera.fov * (Math.PI / 180);
          const cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5;
          camera.position.set(0, 0, cameraZ);
        }

        // Create custom shader material
        material = new THREE.ShaderMaterial({
          uniforms: {
            pointSize: { value: 1.25 },
          },
          vertexShader: `
            uniform float pointSize;
            varying vec3 vColor;
            void main() {
              vColor = color;
              gl_PointSize = pointSize;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
          fragmentShader: `
            varying vec3 vColor;
            void main() {
              gl_FragColor = vec4(vColor, 1.0);
            }
          `,
          vertexColors: true,
        });

        points = new THREE.Points(geometry, material);
        scene.add(points);
        setIsLoading(false);
      },
      undefined,
      (err) => {
        console.error('Error loading PLY file:', err);
        setError('Failed to load 3D point cloud');
        setIsLoading(false);
      }
    );

    // Handle resize
    const handleResize = () => {
      if (!container) return;
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // Animation loop
    let rafId: number;
    const animate = () => {
      rafId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', handleResize);

      if (points) {
        scene.remove(points);
      }
      if (geometry) {
        geometry.dispose();
      }
      if (material) {
        material.dispose();
      }
      renderer.dispose();
      controls.dispose();

      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [plyPath]);

  const handleZoomIn = () => {
    if (cameraRef.current) {
      cameraRef.current.position.multiplyScalar(0.8);
    }
  };

  const handleZoomOut = () => {
    if (cameraRef.current) {
      cameraRef.current.position.multiplyScalar(1.25);
    }
  };

  const handleResetView = () => {
    if (cameraRef.current) {
      cameraRef.current.position.set(0, 0, 100);
    }
  };

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full rounded-lg" />

      {/* Zoom controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <Button
          onClick={handleZoomIn}
          size="sm"
          variant="secondary"
          className="h-8 w-8 p-0"
          title="Zoom in"
        >
          +
        </Button>
        <Button
          onClick={handleZoomOut}
          size="sm"
          variant="secondary"
          className="h-8 w-8 p-0"
          title="Zoom out"
        >
          −
        </Button>
        <Button
          onClick={handleResetView}
          size="sm"
          variant="secondary"
          className="h-8 w-8 p-0"
          title="Reset view"
        >
          ⟲
        </Button>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 rounded-lg">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-2" />
            <p className="text-sm text-gray-600">Loading 3D point cloud...</p>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 rounded-lg">
          <div className="text-center text-red-600">
            <p className="font-medium">{error}</p>
          </div>
        </div>
      )}

      {/* Instructions */}
      {!isLoading && !error && (
        <div className="absolute bottom-4 left-4 bg-black/50 text-white px-3 py-1 rounded text-xs">
          Drag to rotate • Scroll to zoom • Right-click to pan
        </div>
      )}
    </div>
  );
}
