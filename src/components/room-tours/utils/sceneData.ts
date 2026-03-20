export interface SceneData {
  id: string;
  name: string;
  displayName: string;
  videoFile: string;
  plyFile: string;
}

export const scenes: SceneData[] = [
  {
    id: 'scene-003',
    name: 'living_room',
    displayName: 'Living Room',
    videoFile: 'scene-003_living_room.mp4',
    plyFile: 'scene-003_living_room.ply',
  },
  {
    id: 'scene-005',
    name: 'bathroom',
    displayName: 'Bathroom 1',
    videoFile: 'scene-005_bathroom.mp4',
    plyFile: 'scene-005_bathroom.ply',
  },
  {
    id: 'scene-009',
    name: 'bathroom',
    displayName: 'Bathroom 2',
    videoFile: 'scene-009_bathroom.mp4',
    plyFile: 'scene-009_bathroom.ply',
  },
  {
    id: 'scene-020',
    name: 'bedroom',
    displayName: 'Bedroom',
    videoFile: 'scene-020_bedroom.mp4',
    plyFile: 'scene-020_bedroom.ply',
  },
];
