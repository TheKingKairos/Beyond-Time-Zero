import { useMemo, useState } from "react";
import { Activity, Filter, Search } from "lucide-react";

import mockPatientsData from "@/mockData.json";
import type { Patient } from "@/components/PatientTable";
import { PatientTable, SEPSIS_SCORE_TONES } from "@/components/PatientTable";
import {
  PatientDetailDialog,
  type PatientDetail,
  type ProtocolItem,
  type TrendPoint,
} from "@/components/PatientDetailDialog";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type RawTrendPoint = Omit<TrendPoint, "sbp" | "dbp"> & Partial<Pick<TrendPoint, "sbp" | "dbp">>;

interface RawMockPatient {
  patientId: string | number;
  priorityRank: number;
  sepsisScore: number;
  hazardRate: number;
  lastVitalTime: number;
  temp: number;
  hr: number;
  location: string;
  resprate?: number;
  sbp?: number;
  dbp?: number;
  trends?: RawTrendPoint[];
}

const PATIENT_NAMES = [
  "James Anderson",
  "Sarah Brown",
  "Michael Carter",
  "Patricia Davis",
  "Robert Evans",
  "Linda Foster",
  "Maria Garcia",
  "David Harris",
  "Jennifer Jackson",
  "William King",
  "Barbara Lopez",
  "Richard Miller",
  "Susan Nelson",
  "Thomas Ortiz",
  "Nancy Parker",
];

const PRIMARY_NURSES = [
  "Johnson, M.",
  "Williams, R.",
  "Davis, K.",
  "Martinez, L.",
  "Anderson, S.",
  "Taylor, J.",
];

const COMORBIDITY_PRESETS: string[][] = [
  ["Type II Diabetes", "CKD Stage 3", "Hypertension"],
  ["COPD", "Atrial Fibrillation"],
  ["Hypertension", "Coronary Artery Disease"],
  ["CHF", "CKD Stage 3", "Obesity (BMI 34)"],
  ["COPD", "Peripheral Vascular Disease"],
  ["Immunosuppression", "Prior MI"],
];

const RISK_PRESETS: Array<PatientDetail["riskBreakdown"]> = [
  { vitalSigns: 38, labValues: 28, comorbidities: 20, ageRisk: 12 },
  { vitalSigns: 35, labValues: 26, comorbidities: 22, ageRisk: 11 },
  { vitalSigns: 32, labValues: 24, comorbidities: 28, ageRisk: 10 },
  { vitalSigns: 40, labValues: 30, comorbidities: 18, ageRisk: 10 },
  { vitalSigns: 37, labValues: 25, comorbidities: 23, ageRisk: 12 },
];

const SEP_PROTOCOL_TEMPLATES: ProtocolItem[][] = [
  [
    { task: "Blood Cultures (2 sets)", status: "done", completedTime: "14:18" },
    { task: "Broad-Spectrum Antibiotics", status: "overdue", dueTime: "15:05" },
    { task: "Lactate Level Drawn", status: "done", completedTime: "14:05" },
    { task: "Fluid Resuscitation (30 mL/kg)", status: "pending", dueTime: "16:00" },
    { task: "Vasopressor Support Prepared", status: "pending", dueTime: "16:30" },
  ],
  [
    { task: "Blood Cultures (2 sets)", status: "done", completedTime: "13:52" },
    { task: "Broad-Spectrum Antibiotics", status: "pending", dueTime: "15:20" },
    { task: "Lactate Level Drawn", status: "done", completedTime: "13:45" },
    { task: "Lactate Redraw (if >2)", status: "pending", dueTime: "16:10" },
    { task: "Fluid Resuscitation (30 mL/kg)", status: "pending", dueTime: "15:55" },
  ],
  [
    { task: "Blood Cultures (2 sets)", status: "done", completedTime: "14:08" },
    { task: "Broad-Spectrum Antibiotics", status: "pending", dueTime: "15:35" },
    { task: "Source Control Consult", status: "pending", dueTime: "15:50" },
    { task: "Lactate Level Drawn", status: "done", completedTime: "13:58" },
    { task: "Vasopressor Support Prepared", status: "overdue", dueTime: "14:50" },
  ],
];

const PRIORITY_LEGEND = [
  { label: "Red", description: "Critical (80-100)", tone: "critical" },
  { label: "Orange", description: "Warning (60-80)", tone: "warning" },
  { label: "Green", description: "Stable (0-60)", tone: "stable" },
] as const;

function generateTrends(
  baseTemp: number,
  baseHR: number,
  baseLactate: number,
  baseSBP: number,
  baseDBP: number
): TrendPoint[] {
  const trends: TrendPoint[] = [];
  for (let index = 0; index < 4; index += 1) {
    const hoursAgo = 3 - index;
    const timeLabel = hoursAgo === 0 ? "Now" : `${hoursAgo}h ago`;
    const varianceFactor = Math.max(1, 3 - index);
    const sbpVariance = Math.random() * 2 - 1;
    const dbpVariance = Math.random() * 1.5 - 0.75;
    trends.push({
      time: timeLabel,
      temp: Number(
        (baseTemp - (Math.random() * 2 - 1) * varianceFactor * 0.25).toFixed(1)
      ),
      heartRate: Math.round(baseHR - (Math.random() * 2 - 1) * varianceFactor * 4),
      lactate: Number(
        Math.max(0.5, baseLactate - (Math.random() * 2 - 1) * varianceFactor * 0.35).toFixed(1)
      ),
      sbp: Math.max(
        70,
        Math.round(baseSBP + hoursAgo * 4 + sbpVariance * 2)
      ),
      dbp: Math.max(
        40,
        Math.round(baseDBP + hoursAgo * 2 + dbpVariance * 2)
      ),
    });
  }
  return trends;
}

function buildPatientDetail(record: RawMockPatient, index: number): PatientDetail {
  const name = PATIENT_NAMES[index % PATIENT_NAMES.length];
  const primaryNurse = PRIMARY_NURSES[index % PRIMARY_NURSES.length];
  const comorbidities = COMORBIDITY_PRESETS[index % COMORBIDITY_PRESETS.length];
  const risk = RISK_PRESETS[index % RISK_PRESETS.length];
  const protocolTemplate = SEP_PROTOCOL_TEMPLATES[index % SEP_PROTOCOL_TEMPLATES.length];
  const latestRecordedLactate = record.trends?.at(-1)?.lactate;
  const baseLactate = latestRecordedLactate ?? 2.6 + (index % 4) * 0.4;
  const baseSBP = record.sbp ?? record.trends?.at(-1)?.sbp ?? 120;
  const baseDBP = record.dbp ?? record.trends?.at(-1)?.dbp ?? 70;

  const rawTrends = record.trends?.length
    ? record.trends
    : generateTrends(record.temp, record.hr, baseLactate, baseSBP, baseDBP);

  const trendSource: TrendPoint[] = rawTrends.map((trend, trendIndex) => {
    const stepsFromEnd = rawTrends.length - 1 - trendIndex;
    const sbpValue =
      trend.sbp ??
      Math.max(70, Math.round(baseSBP + stepsFromEnd * 4));
    const dbpValue =
      trend.dbp ??
      Math.max(40, Math.round(baseDBP + stepsFromEnd * 2));

    return {
      ...trend,
      sbp: sbpValue,
      dbp: dbpValue,
    };
  });

  return {
    id: String(record.patientId),
    name,
    location: record.location,
    primaryNurse,
    priorityRank: record.priorityRank,
    severityScore: record.sepsisScore,
    hazardRate: record.hazardRate,
    timeSinceLastVital: record.lastVitalTime,
    temperature: record.temp,
    heartRate: record.hr,
    respiratoryRate: record.resprate ?? 18 + (index % 5) * 2,
    trends: trendSource.map((trend) => ({ ...trend })),
    bloodPressure: { systolic: Math.round(baseSBP), diastolic: Math.round(baseDBP) },
    riskBreakdown: { ...risk },
    comorbidities: [...comorbidities],
    sepProtocol: protocolTemplate.map((item) => ({ ...item })),
  };
}

function buildPatientDetails(data: RawMockPatient[]): PatientDetail[] {
  return data.map((record, index) => buildPatientDetail(record, index));
}

export default function App() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedNurse, setSelectedNurse] = useState("all");
  const [selectedPatient, setSelectedPatient] = useState<PatientDetail | null>(null);

  const allPatients = useMemo(
    () => buildPatientDetails((mockPatientsData as RawMockPatient[]) ?? []),
    []
  );

  const nurses = useMemo(
    () => Array.from(new Set(allPatients.map((p) => p.primaryNurse))).sort(),
    [allPatients]
  );

  const filteredPatients = useMemo(() => {
    return allPatients.filter((patient) => {
      const matchesSearch =
        searchTerm === "" ||
        patient.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        patient.name.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesNurse =
        selectedNurse === "all" || patient.primaryNurse === selectedNurse;
      return matchesSearch && matchesNurse;
    });
  }, [allPatients, searchTerm, selectedNurse]);

  const criticalAlerts = useMemo(
    () => filteredPatients.filter((patient) => patient.timeSinceLastVital > 60).length,
    [filteredPatients]
  );

  const handlePatientClick = (patient: Patient) => {
    const detail = allPatients.find((p) => p.id === patient.id) ?? null;
    setSelectedPatient(detail);
  };

  const handleCloseDialog = () => {
    setSelectedPatient(null);
  };

  if (selectedPatient) {
    return <PatientDetailDialog patient={selectedPatient} onClose={handleCloseDialog} />;
  }

  return (
    <div className="min-h-screen bg-slate-100 px-6 py-8 text-slate-900">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <div className="flex flex-col justify-between gap-6 rounded-3xl bg-white px-8 py-6 shadow-sm md:flex-row md:items-start">
          <div>
            <div className="mb-3 flex items-center gap-3">
              <Activity className="h-8 w-8 text-rose-600" />
              <h1 className="text-sm font-semibold uppercase tracking-[0.25em] text-rose-600">
                Emory ED Sepsis Surveillance
              </h1>
            </div>
            <p className="text-sm text-slate-500">
              Cox Survival Hazard Rate Priority · Real-time ML Monitoring
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-xs uppercase tracking-wide text-slate-400">Active Patients</div>
              <div className="font-mono text-2xl text-slate-900">{filteredPatients.length}</div>
            </div>
            {criticalAlerts > 0 ? (
              <Badge variant="destructive" className="py-2">
                {criticalAlerts} Vital Alerts
              </Badge>
            ) : (
              <Badge variant="outline" className="py-2 text-slate-500">
                Stable
              </Badge>
            )}
          </div>
        </div>

        <div className="flex items-center gap-4 rounded-3xl border border-slate-200 bg-white px-6 py-4 shadow-sm">
          <Filter className="h-5 w-5 text-slate-400" />
          <div className="grid w-full gap-4 md:grid-cols-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <Input
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Search Patient ID or Name..."
                className="h-12 rounded-2xl border-slate-200 bg-slate-50 pl-10 text-sm focus:bg-white"
              />
            </div>
            <Select value={selectedNurse} onValueChange={setSelectedNurse}>
              <SelectTrigger className="h-12 rounded-2xl border-slate-200 bg-slate-50 text-sm font-medium text-slate-700 focus:bg-white">
                <SelectValue placeholder="All Patients" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Patients</SelectItem>
                {nurses.map((nurse) => (
                  <SelectItem key={nurse} value={nurse}>
                    {nurse}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <PatientTable patients={filteredPatients} onPatientClick={handlePatientClick} />

        <div className="rounded-3xl border border-slate-200 bg-white px-6 py-4 shadow-sm">
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
                Priority Rank (Severity-Based)
              </h3>
              <div className="flex items-center gap-4 text-sm text-slate-600">
                {PRIORITY_LEGEND.map(({ label, description, tone }) => (
                  <div className="flex items-center gap-2" key={tone}>
                    <span
                      className="inline-flex h-7 w-12 items-center justify-center rounded border px-3 font-mono text-xs uppercase tracking-wide"
                      style={SEPSIS_SCORE_TONES[tone]}
                    >
                      {label}
                    </span>
                    <span>{description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
