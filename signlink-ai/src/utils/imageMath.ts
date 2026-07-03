export const HAND_PADDING = 30;
export const INPUT_SIZE = 224;
export const PREDICTION_THRESHOLD = 0.45;

export interface RoiParams {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Tính toán ROI chuẩn dựa trên landmarks tay.
 */
export function getHandRoi(landmarks: any[], videoWidth: number, videoHeight: number): RoiParams {
  let minX = videoWidth;
  let minY = videoHeight;
  let maxX = 0;
  let maxY = 0;

  for (const lm of landmarks) {
    const x = lm.x * videoWidth;
    const y = lm.y * videoHeight;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const x1 = Math.max(0, Math.floor(minX - HAND_PADDING));
  const y1 = Math.max(0, Math.floor(minY - HAND_PADDING));
  const x2 = Math.min(videoWidth, Math.ceil(maxX + HAND_PADDING));
  const y2 = Math.min(videoHeight, Math.ceil(maxY + HAND_PADDING));

  return {
    x: x1,
    y: y1,
    width: Math.max(1, x2 - x1),
    height: Math.max(1, y2 - y1),
  };
}
