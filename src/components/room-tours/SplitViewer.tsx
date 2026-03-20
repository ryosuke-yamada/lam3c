import type { SceneData } from "./utils/sceneData";
import VideoPlayer from "./VideoPlayer";
import PointCloudViewer from "./PointCloudViewer";

const BASE_URL = import.meta.env.BASE_URL || "/";

interface SplitViewerProps {
  scene: SceneData;
}

export default function SplitViewer({ scene }: SplitViewerProps) {
  const videoPath = `${BASE_URL}videos/${scene.videoFile}`;
  const plyPath = `${BASE_URL}point_clouds/${scene.plyFile}`;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Video Player */}
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-gray-700">RGB Video</h3>
        <div className="relative rounded-xl overflow-hidden border shadow-sm bg-black min-h-[400px] md:min-h-[600px]">
          <VideoPlayer videoPath={videoPath} />
        </div>
      </div>

      {/* Point Cloud Viewer */}
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-gray-700">
          Video-Generated Point Cloud (VGPC)
        </h3>
        <div className="relative rounded-xl overflow-hidden border shadow-sm bg-white min-h-[400px] md:min-h-[600px]">
          <PointCloudViewer plyPath={plyPath} />
        </div>
      </div>
    </div>
  );
}
