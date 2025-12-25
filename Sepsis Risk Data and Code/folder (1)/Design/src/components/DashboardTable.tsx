import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  CONDITION_LABELS,
  getConditionBand,
  SEPSIS_SCORE_TONES,
} from "@/components/PatientTable";

export interface DashboardPatient {
  patientId: string;
  priorityRank: number;
  sepsisScore: number;
  hazardRate: number;
  lastVitalTime: number;
  temp: number;
  hr: number;
  location: string;
}

interface DashboardTableProps {
  patients: DashboardPatient[];
  onPatientClick?: (patient: DashboardPatient) => void;
  selectedPatientId?: string | null;
}

export function DashboardTable({ patients, onPatientClick, selectedPatientId }: DashboardTableProps) {
  const getLastVitalTone = (minutes: number) => {
    if (minutes >= 90) return "text-rose-500 font-semibold";
    if (minutes >= 60) return "text-amber-500 font-semibold";
    return "text-slate-500";
  };

  return (
    <div className="overflow-hidden rounded-3xl border border-slate-200">
      <Table>
        <TableHeader>
          <TableRow className="border-b border-slate-200 bg-slate-100/80 text-xs uppercase tracking-wide text-slate-500">
            <TableHead className="w-[110px] px-6 py-4">Priority Rank</TableHead>
            <TableHead className="w-[130px] px-6 py-4">Condition Score</TableHead>
            <TableHead className="w-[120px] px-6 py-4">Patient ID</TableHead>
            <TableHead className="w-[120px] px-6 py-4">Time Since Last Vital</TableHead>
            <TableHead className="w-[140px] px-6 py-4">Location</TableHead>
            <TableHead className="w-[110px] px-6 py-4">Temp (°F)</TableHead>
            <TableHead className="w-[110px] px-6 py-4">HR (bpm)</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {patients.map((patient, index) => {
            const isSelected = selectedPatientId === patient.patientId;
            const rowBackground = index % 2 === 0 ? "bg-white" : "bg-slate-50/60";
            const conditionBand = getConditionBand(patient.sepsisScore);
            return (
              <TableRow
                key={patient.patientId}
                data-state={isSelected ? "selected" : undefined}
                className={`${rowBackground} cursor-pointer border-b border-slate-200 text-sm text-slate-700 transition hover:bg-slate-100 data-[state=selected]:bg-sky-100`}
                onClick={() => onPatientClick?.(patient)}
              >
                <TableCell className="px-6 py-4">
                  <div className="font-mono text-xs uppercase tracking-wide text-slate-500">#{patient.priorityRank}</div>
                </TableCell>
                <TableCell className="px-6 py-4">
                  <span
                    className="inline-flex min-w-[84px] items-center justify-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-wide"
                    style={SEPSIS_SCORE_TONES[conditionBand]}
                  >
                    {CONDITION_LABELS[conditionBand]}
                  </span>
                </TableCell>
                <TableCell className="px-6 py-4 font-mono tabular-nums text-slate-700">{patient.patientId}</TableCell>
                <TableCell className={`px-6 py-4 font-mono tabular-nums ${getLastVitalTone(patient.lastVitalTime)}`}>
                  {patient.lastVitalTime} min
                </TableCell>
                <TableCell className="px-6 py-4 font-mono text-slate-600">{patient.location}</TableCell>
                <TableCell className="px-6 py-4 font-mono tabular-nums text-slate-700">
                  {patient.temp.toFixed(1)}
                </TableCell>
                <TableCell className="px-6 py-4 font-mono tabular-nums text-slate-700">{patient.hr}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
