export const TREND_Y_RANGES = {
  temperature: { min: 96, max: 106 },
  heartRate: { min: 60, max: 180 },
  lactate: { min: 0, max: 6 },
  bloodPressure: { min: 40, max: 170 },
} as const;

export type TrendRangeKey = keyof typeof TREND_Y_RANGES;
