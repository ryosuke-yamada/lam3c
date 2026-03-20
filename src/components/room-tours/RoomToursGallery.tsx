import { useState } from "react";
import { scenes } from "./utils/sceneData";
import SceneSelector from "./SceneSelector";
import SplitViewer from "./SplitViewer";

export default function RoomToursGallery() {
  const [currentSceneId, setCurrentSceneId] = useState(scenes[0].id);
  const currentScene = scenes.find((s) => s.id === currentSceneId);

  if (!currentScene) {
    return <div>Error: Scene not found</div>;
  }

  return (
    <div className="space-y-6">
      <p className="text-gray-600 leading-relaxed">
        RoomTours is a large-scale dataset of 49,219 indoor scenes generated
        from unlabeled room-walkthrough videos. Each scene contains RGB video
        recordings and their corresponding video-generated point clouds (VGPC)
        reconstructed using off-the-shelf methods.
      </p>
      <SceneSelector
        scenes={scenes}
        currentSceneId={currentSceneId}
        onSceneChange={setCurrentSceneId}
      />
      <SplitViewer scene={currentScene} />
    </div>
  );
}
