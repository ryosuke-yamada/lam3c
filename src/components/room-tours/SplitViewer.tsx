import type { SceneData } from './utils/sceneData';
import VideoPlayer from './VideoPlayer';
import PointCloudViewer from './PointCloudViewer';

const BASE_URL = import.meta.env.BASE_URL || '/';

interface SplitViewerProps {
  scene: SceneData;
}

export default function SplitViewer({ scene }: SplitViewerProps) {
  const videoPath = `${BASE_URL}videos/${scene.videoFile}`;
  const plyPath = `${BASE_URL}point_clouds/${scene.plyFile}`;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 min-h-[400px] md:min-h-[600px]">
      {/* Video Player */}
      <div className="relative rounded-xl overflow-hidden border shadow-sm bg-black">
        <VideoPlayer videoPath={videoPath} />
      </div>

      {/* Point Cloud Viewer */}
      <div className="relative rounded-xl overflow-hidden border shadow-sm bg-white">
        <PointCloudViewer plyPath={plyPath} />
      </div>
    </div>
  );
}
