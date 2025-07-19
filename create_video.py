
import cv2
import numpy as np

def create_dog_drawing(width, height):
    """Creates a simple drawing of a dog on a white background."""
    img = np.full((height, width, 3), 255, dtype=np.uint8) # White background
    
    # Body (brown ellipse)
    cv2.ellipse(img, (width // 2, height // 2 + 50), (100, 40), 0, 0, 360, (139, 69, 19), -1)
    
    # Head (brown circle)
    cv2.circle(img, (width // 2 - 80, height // 2 - 20), 50, (139, 69, 19), -1)
    
    # Eyes (black circles)
    cv2.circle(img, (width // 2 - 95, height // 2 - 30), 5, (0, 0, 0), -1)
    cv2.circle(img, (width // 2 - 65, height // 2 - 30), 5, (0, 0, 0), -1)
    
    # Nose (black circle)
    cv2.circle(img, (width // 2 - 80, height // 2), 8, (0, 0, 0), -1)
    
    # Ears (brown ellipses)
    cv2.ellipse(img, (width // 2 - 110, height // 2 - 60), (20, 40), 30, 0, 360, (139, 69, 19), -1)
    cv2.ellipse(img, (width // 2 - 50, height // 2 - 60), (20, 40), -30, 0, 360, (139, 69, 19), -1)
    
    # Tail (brown line)
    cv2.line(img, (width // 2 + 90, height // 2 + 30), (width // 2 + 130, height // 2), (139, 69, 19), 10)
    
    return img

def create_video_with_dog(output_path='dog_video.mp4', duration=60, fps=30, width=640, height=480):
    """
    Creates a video with a black background and inserts a dog image at specific frames.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration * fps
    dog_frame_1 = 3 * fps   # 3-second mark
    dog_frame_2 = 30 * fps  # 30-second mark

    print(f"Creating video: {output_path} ({total_frames} frames)")
    dog_image = create_dog_drawing(width, height)

    for i in range(total_frames):
        if i == dog_frame_1 or i == dog_frame_2:
            frame = dog_image
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8) # Black frame
        
        video_writer.write(frame)

        if (i + 1) % fps == 0:
            print(f"Processed {i+1}/{total_frames} frames ({ (i+1)//fps } seconds)")

    video_writer.release()
    print(f"Video saved successfully to {output_path}")

if __name__ == "__main__":
    create_video_with_dog()
