import { useRef, useState } from 'react';

interface VideoPlayerProps {
  videoPath: string;
}

export default function VideoPlayer({ videoPath }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(true);

  const handleClick = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="relative w-full h-full">
      <video
        ref={videoRef}
        src={videoPath}
        autoPlay
        muted
        loop
        playsInline
        onClick={handleClick}
        className="w-full h-full object-cover rounded-lg cursor-pointer"
      />
      <div className="absolute bottom-4 left-4 bg-black/50 text-white px-3 py-1 rounded text-sm pointer-events-none">
        {isPlaying ? 'Playing (click to pause)' : 'Paused (click to play)'}
      </div>
    </div>
  );
}
