import { useMemo, useState } from "react";

import type { DashboardPatient } from "@/components/DashboardTable";
import type { TrendPoint } from "@/components/PatientDetailDialog";
import {
  CONDITION_LABELS,
  getConditionBand,
  SEPSIS_SCORE_TONES,
} from "@/components/PatientTable";
import { cn } from "@/lib/utils";
import mockPatientsData from "@/mockData.json";
import { TREND_Y_RANGES } from "@/constants/trendRanges";
import { getComorbidityTone } from "@/constants/comorbidityTones";
import { TrendLineChart } from "@/components/charts/TrendLineChart";
import { AlertTriangle, ArrowLeft } from "lucide-react";

type TrendDirection = "up" | "down";

interface TrendMetric {
  label: string;
  value: string;
  delta: string;
  deltaDirection: TrendDirection;
  range: string;
  sparkline: number[];
}

interface RiskFactor {
  label: string;
  value: number;
  barColor: string;
  annotation: string;
}

type ProtocolStatus = "overdue" | "in-progress" | "scheduled" | "completed";

interface SepProtocolItem {
  title: string;
  status: ProtocolStatus;
  description: string;
  clinician: string;
  timestamp: string;
}

interface PatientDetails {
  trends: TrendMetric[];
  riskBreakdown: RiskFactor[];
  comorbidities: string[];
  sepProtocol: SepProtocolItem[];
}

const CHANGE_THRESHOLDS = {
  temperature: { low: -1.9616567043755875, high: 1.9514529180001021 },
  heartRate: { low: -27.289048822426164, high: 23.162605281867712 },
  respiratoryRate: { low: -6.650194378984333, high: 6.612758101027283 },
  sbp: { low: -37.29654537790982, high: 32.923465253370296 },
  dbp: { low: -34.50440499699454, high: 31.002955284431405 },
  lactate: { low: -2.878780895408627, high: 1.8656539802599306 },
} as const;

function getLatestChange<T extends Record<string, number | string>>(series: T[], key: keyof T) {
  if (series.length < 2) {
    return null;
  }
  const last = series.at(-1)?.[key];
  const prev = series.at(-2)?.[key];
  if (typeof last !== "number" || typeof prev !== "number") {
    return null;
  }
  return last - prev;
}

function buildAlert(delta: number | null, bounds: { low: number; high: number }, label: string) {
  if (delta == null) {
    return null;
  }
  if (delta > bounds.high) {
    return { direction: "rise" as const, delta, label: `Rapid rise in ${label}` };
  }
  if (delta < bounds.low) {
    return { direction: "drop" as const, delta, label: `Sharp drop in ${label}` };
  }
  return null;
}


type PartialBPTrend = Omit<TrendPoint, "sbp" | "dbp"> & Partial<Pick<TrendPoint, "sbp" | "dbp">>;

interface MockPatientRecord {
  patientId: string | number;
  trends?: PartialBPTrend[];
  temp: number;
  hr: number;
  sbp?: number;
  dbp?: number;
}

const patientDetailData: Record<string, PatientDetails> = {
  "P-1001": {
    trends: [
      {
        label: "Lactate",
        value: "4.2 mmol/L",
        delta: "+0.6",
        deltaDirection: "up",
        range: "Last 6 hr",
        sparkline: [2.9, 3.1, 3.4, 3.7, 4, 4.2],
      },
      {
        label: "MAP",
        value: "63 mmHg",
        delta: "-4",
        deltaDirection: "down",
        range: "Last 6 hr",
        sparkline: [71, 69, 68, 66, 64, 63],
      },
      {
        label: "Heart Rate",
        value: "128 bpm",
        delta: "+8",
        deltaDirection: "up",
        range: "Last 6 hr",
        sparkline: [104, 109, 114, 119, 123, 128],
      },
      {
        label: "Temp",
        value: "101.3 °F",
        delta: "+1.1",
        deltaDirection: "up",
        range: "Last 6 hr",
        sparkline: [99.9, 100.1, 100.6, 100.9, 101.1, 101.3],
      },
    ],
    riskBreakdown: [
      {
        label: "Organ Dysfunction",
        value: 78,
        barColor: "bg-red-500",
        annotation: "Creatinine and bilirubin remain elevated from baseline.",
      },
      {
        label: "Source Infection",
        value: 68,
        barColor: "bg-amber-500",
        annotation: "Chest imaging consistent with pneumonia; cultures pending.",
      },
      {
        label: "Hemodynamics",
        value: 59,
        barColor: "bg-blue-500",
        annotation: "Hypotensive despite 1L fluid resuscitation.",
      },
    ],
    comorbidities: ["Type II Diabetes", "Chronic Kidney Disease", "Hypertension"],
    sepProtocol: [
      {
        title: "Broad-Spectrum Antibiotics",
        status: "overdue",
        description: "STAT order waiting on pharmacy verification; escalate immediately.",
        clinician: "Dr. Singh",
        timestamp: "Due 45 min ago",
      },
      {
        title: "Fluid Resuscitation",
        status: "in-progress",
        description: "Second 500 mL NS bolus infusing; reassess MAP after completion.",
        clinician: "RN Mateo",
        timestamp: "Started 12 min ago",
      },
      {
        title: "Vasopressor Support",
        status: "scheduled",
        description: "Norepinephrine prepared if MAP < 65 following fluids.",
        clinician: "Dr. Singh",
        timestamp: "Prep 10 min",
      },
      {
        title: "Lactate Re-draw",
        status: "scheduled",
        description: "Repeat once hemodynamics stabilize.",
        clinician: "Lab",
        timestamp: "In 45 min",
      },
    ],
  },
  "P-1002": {
    trends: [
      {
        label: "Lactate",
        value: "3.7 mmol/L",
        delta: "-0.5",
        deltaDirection: "down",
        range: "Last 6 hr",
        sparkline: [4.4, 4.2, 4.1, 3.9, 3.8, 3.7],
      },
      {
        label: "MAP",
        value: "71 mmHg",
        delta: "+3",
        deltaDirection: "up",
        range: "Last 6 hr",
        sparkline: [60, 63, 65, 68, 70, 71],
      },
      {
        label: "Heart Rate",
        value: "122 bpm",
        delta: "-5",
        deltaDirection: "down",
        range: "Last 6 hr",
        sparkline: [138, 133, 130, 128, 124, 122],
      },
      {
        label: "Temp",
        value: "100.6 °F",
        delta: "-0.4",
        deltaDirection: "down",
        range: "Last 6 hr",
        sparkline: [101.5, 101.2, 101, 100.9, 100.7, 100.6],
      },
    ],
    riskBreakdown: [
      {
        label: "Organ Dysfunction",
        value: 62,
        barColor: "bg-red-400",
        annotation: "Renal function improving; urine output normalized.",
      },
      {
        label: "Source Infection",
        value: 74,
        barColor: "bg-amber-500",
        annotation: "Blood cultures positive for gram-negative rods.",
      },
      {
        label: "Hemodynamics",
        value: 53,
        barColor: "bg-blue-400",
        annotation: "Responding to fluids; MAP trending upward.",
      },
    ],
    comorbidities: ["COPD", "Obesity (BMI 34)", "Hyperlipidemia"],
    sepProtocol: [
      {
        title: "Broad-Spectrum Antibiotics",
        status: "in-progress",
        description: "Piperacillin/tazobactam infusing; monitor for reactions.",
        clinician: "RN Jackson",
        timestamp: "Started 18 min ago",
      },
      {
        title: "Fluid Resuscitation",
        status: "completed",
        description: "30 mL/kg bolus complete; continue to monitor fluid status.",
        clinician: "RN Jackson",
        timestamp: "Finished 5 min ago",
      },
      {
        title: "Source Control",
        status: "scheduled",
        description: "CT chest scheduled to assess for abscess formation.",
        clinician: "Radiology",
        timestamp: "In 25 min",
      },
      {
        title: "Lactate Re-draw",
        status: "scheduled",
        description: "Add to next lab draw; target within 60 minutes.",
        clinician: "Lab",
        timestamp: "In 1 hr",
      },
    ],
  },
  "P-1003": {
    trends: [
      {
        label: "Lactate",
        value: "2.8 mmol/L",
        delta: "-0.3",
        deltaDirection: "down",
        range: "Last 6 hr",
        sparkline: [3.3, 3.2, 3.1, 3, 2.9, 2.8],
      },
      {
        label: "MAP",
        value: "69 mmHg",
        delta: "+2",
        deltaDirection: "up",
        range: "Last 6 hr",
        sparkline: [64, 65, 66, 67, 68, 69],
      },
      {
        label: "Heart Rate",
        value: "130 bpm",
        delta: "+4",
        deltaDirection: "up",
        range: "Last 6 hr",
        sparkline: [118, 121, 124, 126, 128, 130],
      },
      {
        label: "Temp",
        value: "102.1 °F",
        delta: "+0.6",
        deltaDirection: "up",
        range: "Last 6 hr",
        sparkline: [100.9, 101.2, 101.4, 101.6, 101.8, 102.1],
      },
    ],
    riskBreakdown: [
      {
        label: "Organ Dysfunction",
        value: 55,
        barColor: "bg-red-400",
        annotation: "Mild AKI; LFTs within normal limits.",
      },
      {
        label: "Source Infection",
        value: 81,
        barColor: "bg-amber-500",
        annotation: "Suspected abdominal source pending imaging.",
      },
      {
        label: "Hemodynamics",
        value: 61,
        barColor: "bg-blue-500",
        annotation: "Persistent tachycardia with borderline MAP.",
      },
    ],
    comorbidities: ["GERD", "Peripheral Vascular Disease", "Active Smoker"],
    sepProtocol: [
      {
        title: "Broad-Spectrum Antibiotics",
        status: "overdue",
        description: "Order signed but not administered; follow up with pharmacy now.",
        clinician: "Dr. Alvarez",
        timestamp: "Due 20 min ago",
      },
      {
        title: "Fluid Resuscitation",
        status: "in-progress",
        description: "Albumin bolus infusing due to hypoalbuminemia.",
        clinician: "RN Lee",
        timestamp: "Running now",
      },
      {
        title: "Abdominal CT",
        status: "scheduled",
        description: "Patient prepped for contrast-enhanced CT for source control.",
        clinician: "Radiology",
        timestamp: "In 35 min",
      },
      {
        title: "Lactate Re-draw",
        status: "scheduled",
        description: "Coordinate with imaging to avoid delays in redraw.",
        clinician: "Lab",
        timestamp: "Post imaging",
      },
    ],
  },
};

const fallbackDetails: PatientDetails = {
  trends: [
    {
      label: "Lactate",
      value: "3.4 mmol/L",
      delta: "+0.2",
      deltaDirection: "up",
      range: "Last 6 hr",
      sparkline: [3, 3.1, 3.2, 3.3, 3.3, 3.4],
    },
    {
      label: "MAP",
      value: "70 mmHg",
      delta: "-1",
      deltaDirection: "down",
      range: "Last 6 hr",
      sparkline: [74, 73, 72, 71, 70, 69],
    },
  ],
  riskBreakdown: [
    {
      label: "Organ Dysfunction",
      value: 60,
      barColor: "bg-red-400",
      annotation: "Monitoring organ function every 6 hours.",
    },
    {
      label: "Source Infection",
      value: 64,
      barColor: "bg-amber-400",
      annotation: "Awaiting culture identification.",
    },
  ],
  comorbidities: ["No additional history recorded"],
  sepProtocol: [
    {
      title: "Broad-Spectrum Antibiotics",
      status: "scheduled",
      description: "Pending pharmacy verification and nursing administration.",
      clinician: "Primary Team",
      timestamp: "Timing TBD",
    },
  ],
};


function renderSparkBars(values: number[], direction: TrendDirection) {
  if (values.length === 0) {
    return null;
  }
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const spread = Math.max(maxValue - minValue, 0.5);

  const activeColor = direction === "up" ? "bg-emerald-500" : "bg-rose-500";
  const baseColor = direction === "up" ? "bg-emerald-200" : "bg-rose-200";

  return values.map((value, index) => {
    const ratio = (value - minValue) / spread;
    const height = Math.round(28 + ratio * 32);
    const barColor = index === values.length - 1 ? activeColor : baseColor;

    return (
      <span
        // eslint-disable-next-line react/no-array-index-key
        key={index}
        className={`w-2 rounded-sm ${barColor}`}
        style={{ height: `${height}px` }}
      />
    );
  });
}

function trendDeltaClass(direction: TrendDirection) {
  return direction === "up" ? "text-emerald-600" : "text-rose-600";
}

export interface PatientDetailPageProps {
  patient: DashboardPatient;
  onBack: () => void;
}

export function PatientDetailPage({ patient, onBack }: PatientDetailPageProps) {
  const detail = patientDetailData[patient.patientId] ?? fallbackDetails;
  const normalizedPatientId = String(patient.patientId);
  const mockPatient = (mockPatientsData as MockPatientRecord[]).find(
    (item) => String(item.patientId) === normalizedPatientId
  );
  const chartData = useMemo<TrendPoint[]>(() => {
    const baseSBP = mockPatient?.sbp ?? 120;
    const baseDBP = mockPatient?.dbp ?? 70;

    if (mockPatient?.trends?.length) {
      return mockPatient.trends.map((trend, index) => {
        const stepsFromEnd = mockPatient.trends.length - 1 - index;
        const sbp = trend.sbp ?? Math.round(baseSBP + stepsFromEnd * 4);
        const dbp = trend.dbp ?? Math.round(baseDBP + stepsFromEnd * 2);
        return {
          ...trend,
          sbp,
          dbp,
        };
      });
    }

    const lactateMetric = detail.trends.find(
      (metric) => metric.label.toLowerCase().includes("lactate")
    );
    const heartMetric = detail.trends.find(
      (metric) => metric.label.toLowerCase().includes("heart rate")
    );
    const tempMetric = detail.trends.find(
      (metric) => metric.label.toLowerCase().includes("temp")
    );

    const seriesLength = Math.max(
      lactateMetric?.sparkline.length ?? 0,
      heartMetric?.sparkline.length ?? 0,
      tempMetric?.sparkline.length ?? 0
    );

    if (seriesLength === 0) {
      return [];
    }

    const pointsToUse = Math.min(4, seriesLength);
    const startIndex = seriesLength - pointsToUse;

    return Array.from({ length: pointsToUse }, (_, index) => {
      const remaining = pointsToUse - index - 1;
      const label = remaining === 0 ? "Now" : `${remaining}h ago`;
      const sparkIndex = startIndex + index;

      return {
        time: label,
        temp:
          tempMetric?.sparkline[sparkIndex] ??
          mockPatient?.temp ??
          patient.temp,
        heartRate:
          heartMetric?.sparkline[sparkIndex] ??
          mockPatient?.hr ??
          patient.hr,
        lactate:
          lactateMetric?.sparkline[sparkIndex] ??
          (lactateMetric
            ? Number.parseFloat(
                lactateMetric.value.replace(/[^\d.]/g, "") || "0"
              )
            : undefined) ??
          0,
        sbp: Math.round(baseSBP + (pointsToUse - index - 1) * 4),
        dbp: Math.round(baseDBP + (pointsToUse - index - 1) * 2),
      };
    });
  }, [detail.trends, mockPatient, patient.hr, patient.temp]);

  const latestChartPoint = chartData.at(-1);

  if (process.env.NODE_ENV !== "production") {
    // Debugging helper to ensure trend series are populated for the detail page.
    // eslint-disable-next-line no-console
    console.debug("Trend data for detail page", patient.patientId, chartData);
  }
  const fallbackTemperature = mockPatient?.temp != null ? mockPatient.temp.toFixed(1) : patient.temp.toFixed(1);
  const fallbackHeartRate = mockPatient?.hr ?? patient.hr;
  const fallbackSbp =
    mockPatient?.sbp ?? (latestChartPoint?.sbp ?? undefined) ?? 120;
  const fallbackDbp =
    mockPatient?.dbp ?? (latestChartPoint?.dbp ?? undefined) ?? 70;
  const conditionBand = getConditionBand(patient.sepsisScore);
  const summaryMetrics = [
    {
      label: "Temperature",
      value: latestChartPoint
        ? `${latestChartPoint.temp.toFixed(1)} °F`
        : `${fallbackTemperature} °F`,
    },
    {
      label: "Heart Rate",
      value: latestChartPoint ? `${latestChartPoint.heartRate} bpm` : `${fallbackHeartRate} bpm`,
    },
    {
      label: "Lactate",
      value: latestChartPoint ? `${latestChartPoint.lactate.toFixed(1)} mmol/L` : "--",
    },
    {
      label: "Blood Pressure",
      value: latestChartPoint
        ? `${latestChartPoint.sbp}/${latestChartPoint.dbp} mmHg`
        : `${Math.round(fallbackSbp)}/${Math.round(fallbackDbp)} mmHg`,
    },
  ];

  const temperatureSeries = chartData.map((point) => ({ time: point.time, value: point.temp }));
  const heartRateSeries = chartData.map((point) => ({ time: point.time, value: point.heartRate }));
  const lactateSeries = chartData.map((point) => ({ time: point.time, value: point.lactate }));
  const bloodPressureSeries = chartData.map((point) => ({ time: point.time, sbp: point.sbp, dbp: point.dbp }));

  const temperatureAlert = buildAlert(
    getLatestChange(temperatureSeries, "value"),
    CHANGE_THRESHOLDS.temperature,
    "temperature"
  );
  const heartRateAlert = buildAlert(
    getLatestChange(heartRateSeries, "value"),
    CHANGE_THRESHOLDS.heartRate,
    "heart rate"
  );
  const lactateAlert = buildAlert(
    getLatestChange(lactateSeries, "value"),
    CHANGE_THRESHOLDS.lactate,
    "lactate"
  );
  const bloodPressureAlert =
    buildAlert(getLatestChange(bloodPressureSeries, "sbp"), CHANGE_THRESHOLDS.sbp, "systolic BP") ??
    buildAlert(getLatestChange(bloodPressureSeries, "dbp"), CHANGE_THRESHOLDS.dbp, "diastolic BP");

  const metricCharts = [
    {
      id: "temperature",
      title: "Temperature",
      unit: "°F",
      data: temperatureSeries,
      valueFormatter: (value: number) => `${value.toFixed(1)} °F`,
      latestValue: latestChartPoint ? `${latestChartPoint.temp.toFixed(1)} °F` : `${patient.temp.toFixed(1)} °F`,
      latestLabel: "Current temperature",
      lines: [
        {
          key: "value",
          name: "Temperature",
          stroke: "#dc2626",
          mutedStroke: "rgba(220, 38, 38, 0.45)",
          gradientFrom: "rgba(220, 38, 38, 0.35)",
          gradientTo: "rgba(248, 113, 113, 0.05)",
          showArea: false,
        },
      ],
      alert: temperatureAlert,
      yRange: TREND_Y_RANGES.temperature,
    },
    {
      id: "heart-rate",
      title: "Heart Rate",
      unit: "bpm",
      data: heartRateSeries,
      valueFormatter: (value: number) => `${Math.round(value)} bpm`,
      latestValue: latestChartPoint ? `${latestChartPoint.heartRate} bpm` : `${patient.hr} bpm`,
      latestLabel: "Current heart rate",
      lines: [
        {
          key: "value",
          name: "Heart Rate",
          stroke: "#f97316",
          mutedStroke: "rgba(249, 115, 22, 0.45)",
          gradientFrom: "rgba(249, 115, 22, 0.3)",
          gradientTo: "rgba(251, 146, 60, 0.05)",
          showArea: false,
        },
      ],
      alert: heartRateAlert,
      yRange: TREND_Y_RANGES.heartRate,
    },
    {
      id: "lactate",
      title: "Lactate",
      unit: "mmol/L",
      data: lactateSeries,
      valueFormatter: (value: number) => `${value.toFixed(1)} mmol/L`,
      latestValue: latestChartPoint ? `${latestChartPoint.lactate.toFixed(1)} mmol/L` : "--",
      latestLabel: "Current lactate",
      lines: [
        {
          key: "value",
          name: "Lactate",
          stroke: "#9333ea",
          mutedStroke: "rgba(147, 51, 234, 0.45)",
          gradientFrom: "rgba(147, 51, 234, 0.3)",
          gradientTo: "rgba(216, 180, 254, 0.05)",
          showArea: false,
        },
      ],
      alert: lactateAlert,
      yRange: TREND_Y_RANGES.lactate,
    },
    {
      id: "blood-pressure",
      title: "Blood Pressure",
      unit: "mmHg",
      data: bloodPressureSeries,
      valueFormatter: (value: number, name?: string) => {
        const label = name === "dbp" ? "DBP" : "SBP";
        return `${label}: ${Math.round(value)} mmHg`;
      },
      latestValue: latestChartPoint
        ? `${latestChartPoint.sbp}/${latestChartPoint.dbp} mmHg`
        : `${Math.round(fallbackSbp)}/${Math.round(fallbackDbp)} mmHg`,
      latestLabel: "Current blood pressure",
      lines: [
        {
          key: "sbp",
          name: "SBP",
          stroke: "#2563eb",
          mutedStroke: "rgba(37, 99, 235, 0.45)",
          gradientFrom: "rgba(37, 99, 235, 0.35)",
          gradientTo: "rgba(191, 219, 254, 0.08)",
        },
        {
          key: "dbp",
          name: "DBP",
          stroke: "#0ea5e9",
          mutedStroke: "rgba(14, 165, 233, 0.45)",
          gradientFrom: "rgba(14, 165, 233, 0.3)",
          gradientTo: "rgba(125, 211, 252, 0.05)",
          strokeDasharray: "6 3",
        },
      ],
      height: 130,
      alert: bloodPressureAlert,
      yRange: TREND_Y_RANGES.bloodPressure,
    },
  ];
  const [selectedTrend, setSelectedTrend] = useState<(typeof metricCharts)[number] | null>(null);
  const criticalAlerts = metricCharts.filter((metric) => metric.alert);

  return (
    <div className="min-h-screen bg-white text-gray-900">
      <header className="border-b border-slate-200 bg-white/90 px-6 py-6 shadow-sm">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-6">
            <div className="flex items-center gap-4">
              <button
                type="button"
                onClick={onBack}
                className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-2"
              >
                <span aria-hidden="true">←</span>
                Back to Dashboard
              </button>
              <span className="rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-700">
                Patient detail
              </span>
            </div>
            <div>
              <h1 className="text-3xl font-semibold text-slate-900">Patient Detail Overview</h1>
              <p className="mt-2 text-sm text-slate-500">
                Integrated clinical signals and risk context for the selected patient.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-6 text-sm text-slate-600 md:grid-cols-3">
              <div>
                <p className="uppercase tracking-wide text-slate-400">Patient ID</p>
                <p className="font-sans text-sm text-slate-900">{patient.patientId}</p>
              </div>
              <div>
                <p className="uppercase tracking-wide text-slate-400">Priority Rank</p>
                <p className="font-semibold text-slate-900">#{patient.priorityRank}</p>
              </div>
              <div>
                <p className="uppercase tracking-wide text-slate-400">Condition Score</p>
                <span
                  className="mt-1 inline-flex min-w-[120px] items-center justify-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide"
                  style={SEPSIS_SCORE_TONES[conditionBand]}
                >
                  {CONDITION_LABELS[conditionBand]}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-600">
              <p className="uppercase tracking-wide text-slate-400">Time Since Last Vital</p>
              <p className="mt-1 font-mono text-sm text-slate-900">{patient.lastVitalTime} min ago</p>
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-center">
                <div className="space-y-1">
                  <p className="uppercase tracking-wide text-slate-400">Temp</p>
                  <p className="font-mono text-sm text-slate-900">{patient.temp.toFixed(1)} °F</p>
                </div>
                <div className="space-y-1">
                  <p className="uppercase tracking-wide text-slate-400">HR</p>
                  <p className="font-mono text-sm text-slate-900">{patient.hr}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl px-6 py-10">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <section className="space-y-8 lg:pr-6">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                Diagnostic Justification
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Trend highlights explaining the current risk tier and priority placement.
              </p>
              {chartData.length > 0 ? (
                <div className="mt-4 space-y-4 rounded-2xl border border-slate-200 bg-white px-4 py-3.5 shadow-sm">
                  <div>
                    <div className="flex flex-wrap items-center gap-3">
                      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                        Vital Trend (last 4 hr)
                      </p>
                      {criticalAlerts.length > 0 ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-rose-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-rose-600">
                          <AlertTriangle className="h-3.5 w-3.5" />
                          {criticalAlerts.length} Alert{criticalAlerts.length > 1 ? "s" : ""}
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-1 text-sm text-slate-500">
                      Focused micro-trends for temperature, heart rate, lactate, and blood pressure.
                    </p>
                    {criticalAlerts.length > 0 ? (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {criticalAlerts.map((metric) => (
                          <span
                            key={metric.id}
                            className="inline-flex items-center gap-1 rounded-full bg-rose-600 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-white"
                          >
                            <span className="inline-block h-1.5 w-1.5 rounded-full bg-white" />
                            {metric.title}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    <div className="mt-3 grid gap-2 sm:grid-cols-2 md:grid-cols-4">
                      {summaryMetrics.map((metric) => (
                        <div key={metric.label} className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                          <p className="text-[11px] uppercase tracking-wide text-slate-400">{metric.label}</p>
                          <p className="font-semibold text-slate-900">{metric.value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  {selectedTrend ? (
                    <div className="mt-3 space-y-6 rounded-3xl border border-slate-200 bg-white px-4 py-4 shadow-sm">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <button
                          type="button"
                          onClick={() => setSelectedTrend(null)}
                          className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500 hover:bg-slate-100"
                        >
                          <ArrowLeft className="h-4 w-4" /> Patient info
                        </button>
                        <div className="text-right">
                          <p className="text-[10px] uppercase tracking-[0.3em] text-slate-400">Metric focus</p>
                          <p className="text-lg font-semibold text-slate-900">{selectedTrend.title}</p>
                          <p className="text-xs text-slate-500">Unit · {selectedTrend.unit}</p>
                        </div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                        <TrendLineChart
                          dataPoints={selectedTrend.data}
                          lines={selectedTrend.lines}
                          height={260}
                          suggestedYRange={selectedTrend.yRange}
                          valueFormatter={(value, datasetLabel) => selectedTrend.valueFormatter(value, datasetLabel)}
                          labelFormatter={(label) => `Time · ${label}`}
                          highlightDirection={selectedTrend.alert?.direction}
                          showPointLabels={selectedTrend.lines.length === 1}
                        />
                      </div>
                      <div className="mt-4 grid gap-3 md:grid-cols-3">
                        <div className="min-w-[200px] flex-1 rounded-2xl border border-slate-200 bg-white p-4">
                          <p className="text-[11px] uppercase tracking-[0.3em] text-slate-400">Current</p>
                          <p className="text-2xl font-semibold text-slate-900">{selectedTrend.latestValue ?? "--"}</p>
                          <p className="text-sm text-slate-500">{selectedTrend.latestLabel ?? "Current value"}</p>
                        </div>
                        <div className="min-w-[200px] flex-1 rounded-2xl border border-slate-200 bg-white p-4">
                          <p className="text-[11px] uppercase tracking-[0.3em] text-slate-400">Status</p>
                          {selectedTrend.alert ? (
                            <p className="text-lg font-semibold text-rose-600">{selectedTrend.alert.label}</p>
                          ) : (
                            <p className="text-lg font-semibold text-emerald-600">Stable</p>
                          )}
                          <p className="text-sm text-slate-500">Auto-monitored</p>
                        </div>
                        <div className="min-w-[200px] flex-1 rounded-2xl border border-slate-200 bg-white p-4">
                          <p className="text-[11px] uppercase tracking-[0.3em] text-slate-400">Samples tracked</p>
                          <p className="text-2xl font-semibold text-slate-900">{selectedTrend.data.length}</p>
                        </div>
                      </div>
                      </div>
                    </div>
                  ) : (
                    <div className="mt-3 grid grid-cols-2 gap-4">
                      {metricCharts.map((metric) => (
                        <article
                          key={metric.id}
                          className={cn(
                            "chart-card relative flex cursor-pointer flex-col overflow-hidden p-3.5 text-slate-900 transition hover:-translate-y-0.5",
                            !metric.alert && "border-slate-200/70 shadow-[0_18px_30px_rgba(15,23,42,0.08)]"
                          )}
                          data-alert={metric.alert?.direction ?? undefined}
                          role="button"
                          tabIndex={0}
                          onClick={() => setSelectedTrend(metric)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              setSelectedTrend(metric);
                            }
                          }}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                                {metric.title} Trend
                              </p>
                              <span className="text-[11px] font-medium text-slate-300">{metric.unit}</span>
                            </div>
                          </div>
                          <div
                            className="chart-alert mt-3 border border-slate-200 bg-white p-2"
                            data-alert={metric.alert?.direction ?? undefined}
                          >
                            <TrendLineChart
                              dataPoints={metric.data}
                              lines={metric.lines}
                              height={metric.height ?? 130}
                              suggestedYRange={metric.yRange}
                              valueFormatter={(value, datasetLabel) => metric.valueFormatter(value, datasetLabel)}
                              labelFormatter={(label) => `Time · ${label}`}
                              highlightDirection={metric.alert?.direction}
                            />
                          </div>
                          <div className="mt-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-left text-slate-900">
                            {metric.alert ? (
                              <p className="alert-delta-chip alert-delta-chip--rise mb-2">{metric.alert.label}</p>
                            ) : null}
                            <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
                              {metric.latestLabel ?? "Current value"}
                            </p>
                            <p className="text-lg font-semibold text-slate-900">{metric.latestValue ?? "--"}</p>
                          </div>
                        </article>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <p className="mt-4 text-xs text-slate-400">No trend data available for charting.</p>
              )}
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                {detail.trends.map((trend) => (
                  <article
                    key={trend.label}
                    className="rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-sm"
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                          {trend.label}
                        </p>
                        <p className="mt-1 text-lg font-semibold text-slate-900">{trend.value}</p>
                        <p className={`mt-1 text-xs font-medium ${trendDeltaClass(trend.deltaDirection)}`}>
                          {trend.deltaDirection === "up" ? "▲" : "▼"} {trend.delta} vs {trend.range}
                        </p>
                      </div>
                    </div>
                    <div className="mt-4 flex h-16 items-end gap-1">
                      {renderSparkBars(trend.sparkline, trend.deltaDirection)}
                    </div>
                  </article>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                Risk Breakdown
              </h3>
              <div className="mt-4 space-y-4">
                {detail.riskBreakdown.map((risk) => (
                  <article
                    key={risk.label}
                    className="rounded-2xl border border-slate-200 bg-slate-50/70 px-5 py-4 shadow-sm"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                          {risk.label}
                        </p>
                        <p className="mt-2 text-sm text-slate-600">{risk.annotation}</p>
                      </div>
                      <span className="font-mono text-lg font-semibold text-slate-900">{risk.value}%</span>
                    </div>
                    <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-200">
                      <div
                        className={`h-full ${risk.barColor}`}
                        style={{ width: `${Math.min(100, Math.max(0, risk.value))}%` }}
                      />
                    </div>
                  </article>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                Comorbidities
              </h3>
              <div className="mt-3 flex flex-wrap gap-2">
                {detail.comorbidities.map((item, index) => (
                  <span
                    key={item}
                    className={cn(
                      "inline-flex max-w-full flex-wrap items-center rounded-full border px-4 py-1.5 text-base font-semibold leading-tight shadow-sm whitespace-normal break-words",
                      getComorbidityTone(index)
                    )}
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          </section>

        </div>
      </main>
    </div>
  );
}
