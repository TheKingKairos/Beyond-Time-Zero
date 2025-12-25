const COMORBIDITY_TONES = [
  "border-rose-300 bg-rose-100 text-rose-800",
  "border-amber-300 bg-amber-100 text-amber-800",
  "border-emerald-300 bg-emerald-100 text-emerald-800",
  "border-sky-300 bg-sky-100 text-sky-800",
  "border-indigo-300 bg-indigo-100 text-indigo-800",
  "border-fuchsia-300 bg-fuchsia-100 text-fuchsia-800",
] as const;

export function getComorbidityTone(index: number) {
  return COMORBIDITY_TONES[((index % COMORBIDITY_TONES.length) + COMORBIDITY_TONES.length) % COMORBIDITY_TONES.length];
}
