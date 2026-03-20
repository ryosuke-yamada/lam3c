import type { SceneData } from "./utils/sceneData";
import { cn } from "@/lib/utils";

interface SceneSelectorProps {
  scenes: SceneData[];
  currentSceneId: string;
  onSceneChange: (sceneId: string) => void;
}

export default function SceneSelector({
  scenes,
  currentSceneId,
  onSceneChange,
}: SceneSelectorProps) {
  return (
    <div className="flex gap-4 flex-wrap justify-center">
      {scenes.map((scene) => (
        <button
          key={scene.id}
          onClick={() => onSceneChange(scene.id)}
          className={cn(
            "flex flex-col items-center gap-2 px-6 py-4 rounded-xl border-2 transition-all",
            currentSceneId === scene.id
              ? "border-blue-500 bg-blue-50 shadow-md"
              : "border-gray-200 hover:border-gray-300 bg-white hover:shadow-sm",
          )}
        >
          <span className="font-medium text-sm">{scene.displayName}</span>
          <span className="text-xs text-gray-500">{scene.id}</span>
        </button>
      ))}
    </div>
  );
}
