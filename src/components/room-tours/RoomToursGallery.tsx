import { useState } from 'react';
import { scenes } from './utils/sceneData';
import SceneSelector from './SceneSelector';
import SplitViewer from './SplitViewer';

export default function RoomToursGallery() {
  const [currentSceneId, setCurrentSceneId] = useState(scenes[0].id);
  const currentScene = scenes.find((s) => s.id === currentSceneId);

  if (!currentScene) {
    return <div>Error: Scene not found</div>;
  }

  return (
    <div className="space-y-6">
      <SceneSelector
        scenes={scenes}
        currentSceneId={currentSceneId}
        onSceneChange={setCurrentSceneId}
      />
      <SplitViewer scene={currentScene} />
    </div>
  );
}
