import type { CSSProperties } from "react";

import { cn } from "@/lib/utils";

export interface Patient {
  id: string;
  name: string;
  location: string;
  primaryNurse: string;
  priorityRank: number;
  severityScore: number;
  hazardRate: number;
  timeSinceLastVital: number;
  temperature: number;
  heartRate: number;
  respiratoryRate: number;
}

interface PatientTableProps {
  patients: Patient[];
  onPatientClick?: (patient: Patient) => void;
}

export type ConditionBand = "critical" | "warning" | "stable";

export const CONDITION_LABELS: Record<ConditionBand, string> = {
  critical: "Critical",
  warning: "Warning",
  stable: "Stable",
};

export const SEPSIS_SCORE_TONES: Record<ConditionBand, CSSProperties> = {
  critical: {
    color: "#991b1b",
    backgroundColor: "#fee2e2",
    borderColor: "#fecaca",
  },
  warning: {
    color: "#92400e",
    backgroundColor: "#fef3c7",
    borderColor: "#fde68a",
  },
  stable: {
    color: "#065f46",
    backgroundColor: "#d1fae5",
    borderColor: "#a7f3d0",
  },
};

export function getConditionBand(score: number): ConditionBand {
  const value = score <= 1 ? score * 100 : score;
  if (value >= 93.6) {
    return "critical";
  }
  if (value <= 35.6) {
    return "stable";
  }
  return "warning";
}

function vitalTone(minutes: number) {
  if (minutes >= 90) return "text-rose-600 font-semibold";
  if (minutes >= 60) return "text-amber-500 font-semibold";
  return "text-slate-500";
}

export function PatientTable({ patients, onPatientClick }: PatientTableProps) {
  return (
    <div className="overflow-hidden rounded-3xl border border-slate-200">
      <table className="w-full">
        <thead className="border-b border-slate-200 bg-slate-100/80 text-left text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-6 py-4">Priority Rank</th>
            <th className="px-6 py-4">Condition Score</th>
            <th className="px-6 py-4">Patient ID</th>
            <th className="px-6 py-4">Patient Name</th>
            <th className="px-6 py-4">Primary Nurse</th>
            <th className="px-6 py-4">Time Since Last Vital</th>
            <th className="px-6 py-4">Temp (°F)</th>
            <th className="px-6 py-4 text-center">HR (bpm)</th>
          </tr>
        </thead>
        <tbody>
          {patients.map((patient, index) => {
            const rowBackground = index % 2 === 0 ? "bg-white" : "bg-slate-50/60";
            const conditionBand = getConditionBand(patient.severityScore);
            return (
              <tr
                key={patient.id}
                className={cn(
                  rowBackground,
                  "cursor-pointer border-b border-slate-200 text-sm text-slate-700 transition hover:bg-slate-100"
                )}
                onClick={() => onPatientClick?.(patient)}
              >
                <td className="px-6 py-4">
                  <span className="font-mono text-sm uppercase tracking-wide text-slate-600">
                    #{patient.priorityRank}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span
                    className="inline-flex min-w-[84px] items-center justify-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-wide"
                    style={SEPSIS_SCORE_TONES[conditionBand]}
                  >
                    {CONDITION_LABELS[conditionBand]}
                  </span>
                </td>
                <td className="px-6 py-4 font-mono tabular-nums text-slate-700">{patient.id}</td>
                <td className="px-6 py-4 text-slate-600">{patient.name}</td>
                <td className="px-6 py-4 text-slate-600">{patient.primaryNurse}</td>
                <td className={cn("px-6 py-4 font-mono tabular-nums", vitalTone(patient.timeSinceLastVital))}>
                  {patient.timeSinceLastVital} min
                </td>
                <td className="px-6 py-4 font-mono tabular-nums text-slate-700">
                  {patient.temperature.toFixed(1)}
                </td>
                <td className="px-6 py-4 text-center font-mono tabular-nums text-slate-700">
                  {patient.heartRate}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
