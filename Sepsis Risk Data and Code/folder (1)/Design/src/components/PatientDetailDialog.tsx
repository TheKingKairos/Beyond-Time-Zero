import { useState, type ReactNode } from "react";
import { AlertTriangle, ArrowLeft } from "lucide-react";

import type { Patient } from "@/components/PatientTable";
import {
  CONDITION_LABELS,
  getConditionBand,
  SEPSIS_SCORE_TONES,
} from "@/components/PatientTable";
import { cn } from "@/lib/utils";
import { TREND_Y_RANGES } from "@/constants/trendRanges";
import { getComorbidityTone } from "@/constants/comorbidityTones";
import { TrendLineChart } from "@/components/charts/TrendLineChart";

export interface TrendPoint {
  time: string;
  temp: number;
  heartRate: number;
  lactate: number;
  sbp: number;
  dbp: number;
}

export interface RiskBreakdown {
  vitalSigns: number;
  labValues: number;
  comorbidities: number;
  ageRisk: number;
}

export interface ProtocolItem {
  task: string;
  status: "done" | "pending" | "overdue";
  completedTime?: string;
  dueTime?: string;
}

export interface PatientDetail extends Patient {
  trends: TrendPoint[];
  bloodPressure: { systolic: number; diastolic: number };
  riskBreakdown: RiskBreakdown;
  comorbidities: string[];
  sepProtocol: ProtocolItem[];
}

interface PatientDetailDialogProps {
  patient: PatientDetail | null;
  onClose: () => void;
}

type TrendDirection = "up" | "down" | "flat";

interface TrendMetricCard {
  label: string;
  value: string;
  delta: string;
  direction: TrendDirection;
  sparkline: number[];
  rangeLabel: string;
}

type MetricChartConfig = {
  id: string;
  title: string;
  unit: string;
  data: Array<{ time: string } & Record<string, number>>;
  lines: Array<{
    key: string;
    name: string;
    stroke: string;
    mutedStroke?: string;
    gradientFrom: string;
    gradientTo: string;
    showArea?: boolean;
    strokeDasharray?: string;
  }>;
  valueFormatter: (value: number, name?: string) => string;
  latestValue: string;
  latestLabel: string;
  alert?: {
    direction: "rise" | "drop";
    delta: number;
    label: string;
  };
  height?: number;
  yRange?: { min: number; max: number };
};

const CHANGE_THRESHOLDS = {
  temperature: { low: -1.9616567043755875, high: 1.9514529180001021 },
  heartRate: { low: -27.289048822426164, high: 23.162605281867712 },
  respiratoryRate: { low: -6.650194378984333, high: 6.612758101027283 },
  sbp: { low: -37.29654537790982, high: 32.923465253370296 },
  dbp: { low: -34.50440499699454, high: 31.002955284431405 },
  lactate: { low: -2.878780895408627, high: 1.8656539802599306 },
} as const;

function getLatestChange<T extends Record<string, number | string>>(
  series: T[],
  key: keyof T
): number | null {
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


export function PatientDetailDialog({ patient, onClose }: PatientDetailDialogProps) {
  if (!patient) {
    return null;
  }

  const [selectedTrend, setSelectedTrend] = useState<MetricChartConfig | null>(null);

  const trendData = patient.trends ?? [];
  const latestTrend = trendData.at(-1);

  if (process.env.NODE_ENV !== "production") {
    // Debugging helper to confirm trend data presence when rendering charts
    // eslint-disable-next-line no-console
    console.debug("Trend data for patient dialog", patient.id, trendData);
  }

  const tempTrend = trendData.map((point) => ({ time: point.time, value: point.temp }));
  const heartTrend = trendData.map((point) => ({ time: point.time, value: point.heartRate }));
  const lactateTrend = trendData.map((point) => ({ time: point.time, value: point.lactate }));
  const bpTrend = trendData.map((point) => ({
    time: point.time,
    sbp: point.sbp,
    dbp: point.dbp,
  }));

  const tempChange = getLatestChange(tempTrend, "value");
  const heartChange = getLatestChange(heartTrend, "value");
  const lactateChange = getLatestChange(lactateTrend, "value");
  const sbpChange = getLatestChange(bpTrend, "sbp");
  const dbpChange = getLatestChange(bpTrend, "dbp");

  const tempAlert = buildAlert(tempChange, CHANGE_THRESHOLDS.temperature, "temperature");
  const heartAlert = buildAlert(heartChange, CHANGE_THRESHOLDS.heartRate, "heart rate");
  const lactateAlert = buildAlert(lactateChange, CHANGE_THRESHOLDS.lactate, "lactate");
  const bloodPressureAlert =
    buildAlert(sbpChange, CHANGE_THRESHOLDS.sbp, "systolic BP") ??
    buildAlert(dbpChange, CHANGE_THRESHOLDS.dbp, "diastolic BP");

  const metricCharts: MetricChartConfig[] = [
    {
      id: "temperature",
      title: "Temperature",
      unit: "°F",
      data: tempTrend,
      valueFormatter: (value) => `${value.toFixed(1)} °F`,
      latestValue: latestTrend ? `${latestTrend.temp.toFixed(1)} °F` : "--",
      latestLabel: "Current temperature",
      alert: tempAlert,
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
      yRange: TREND_Y_RANGES.temperature,
    },
    {
      id: "heart-rate",
      title: "Heart Rate",
      unit: "bpm",
      data: heartTrend,
      valueFormatter: (value) => `${Math.round(value)} bpm`,
      latestValue: latestTrend ? `${latestTrend.heartRate} bpm` : "--",
      latestLabel: "Current heart rate",
      alert: heartAlert,
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
      yRange: TREND_Y_RANGES.heartRate,
    },
    {
      id: "lactate",
      title: "Lactate",
      unit: "mmol/L",
      data: lactateTrend,
      valueFormatter: (value) => `${value.toFixed(1)} mmol/L`,
      latestValue: latestTrend ? `${latestTrend.lactate.toFixed(1)} mmol/L` : "--",
      latestLabel: "Current lactate",
      alert: lactateAlert,
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
      yRange: TREND_Y_RANGES.lactate,
    },
    {
      id: "blood-pressure",
      title: "Blood Pressure",
      unit: "mmHg",
      data: bpTrend,
      valueFormatter: (value, name) => `${name === "dbp" ? "DBP" : "SBP"}: ${Math.round(value)} mmHg`,
      latestValue: latestTrend
        ? `${latestTrend.sbp}/${latestTrend.dbp} mmHg`
        : `${patient.bloodPressure.systolic}/${patient.bloodPressure.diastolic} mmHg`,
      latestLabel: "Current blood pressure",
      alert: bloodPressureAlert,
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
      yRange: TREND_Y_RANGES.bloodPressure,
    },
];

const criticalAlerts = metricCharts.filter((metric) => metric.alert);

  const conditionBand = getConditionBand(patient.severityScore);
  const summaryDetails: Array<{ label: string; value: ReactNode }> = [
    { label: "Patient ID", value: patient.id },
    { label: "Priority Rank", value: `#${patient.priorityRank}` },
    { label: "Primary Nurse", value: patient.primaryNurse },
    { label: "Location", value: patient.location },
    { label: "Time Since Last Vital", value: `${patient.timeSinceLastVital} min` },
    {
      label: "Condition Score",
      value: (
        <span
          className="inline-flex min-w-[120px] items-center justify-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-wide"
          style={SEPSIS_SCORE_TONES[conditionBand]}
        >
          {CONDITION_LABELS[conditionBand]}
        </span>
      ),
    },
  ];

  return (
    <div className="bg-slate-50 text-slate-900">
      <div className="mx-auto flex max-w-5xl flex-col gap-3 px-5 py-4">
        <header className="flex flex-col gap-4 rounded-3xl border border-slate-200 bg-white px-4 py-3 shadow-sm md:flex-row md:items-start md:justify-between md:gap-8">
          <div className="flex flex-1 items-center gap-3 md:min-w-0">
            <button
              type="button"
              onClick={onClose}
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm transition hover:bg-slate-100"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </button>
            <div className="space-y-0.5">
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-rose-600">
                Patient Snapshot
              </p>
              <h1 className="text-xl font-semibold text-slate-900">{patient.name}</h1>
            </div>
          </div>
          <div className="flex w-full flex-wrap gap-2 sm:justify-start md:w-auto md:flex-col md:items-end">
            {summaryDetails.map((item) => (
              <InfoChip
                key={item.label}
                label={item.label}
                value={item.value}
                className="flex-1 text-left sm:flex-none sm:min-w-[220px] md:w-[200px] md:text-right"
              />
            ))}
          </div>
        </header>

        <main className="grid flex-1 gap-3 lg:grid-cols-[1.8fr_1fr]">
          {selectedTrend ? (
            <section className="rounded-3xl border border-slate-200 bg-white px-4 py-4 shadow-lg lg:col-span-2">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <button
                  type="button"
                  onClick={() => setSelectedTrend(null)}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500 hover:bg-slate-100"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Patient info
                </button>
                <div className="text-right">
                  <p className="text-[10px] uppercase tracking-[0.3em] text-slate-400">Metric focus</p>
                  <h2 className="text-xl font-semibold text-slate-900">{selectedTrend.title}</h2>
                  <p className="text-xs text-slate-500">Unit · {selectedTrend.unit}</p>
                </div>
              </div>
              <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
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
                <div className="min-w-[180px] flex-1 rounded-2xl border border-slate-200 bg-white p-4">
                  <p className="text-[11px] uppercase tracking-[0.3em] text-slate-400">Current</p>
                  <p className="text-2xl font-semibold text-slate-900">{selectedTrend.latestValue}</p>
                  <p className="text-sm text-slate-500">{selectedTrend.latestLabel}</p>
                </div>
                <div className="min-w-[180px] flex-1 rounded-2xl border border-slate-200 bg-white p-4">
                  <p className="text-[11px] uppercase tracking-[0.3em] text-slate-400">Status</p>
                  {selectedTrend.alert ? (
                    <p className="text-lg font-semibold text-rose-600">{selectedTrend.alert.label}</p>
                  ) : (
                    <p className="text-lg font-semibold text-emerald-600">Stable</p>
                  )}
                  <p className="text-sm text-slate-500">Auto-monitored</p>
                </div>
                <div className="min-w-[180px] flex-1 rounded-2xl border border-slate-200 bg-white p-4">
                  <p className="text-[11px] uppercase tracking-[0.3em] text-slate-400">Samples tracked</p>
                  <p className="text-2xl font-semibold text-slate-900">{selectedTrend.data.length}</p>
                </div>
              </div>
            </section>
          ) : (
          <section className="rounded-3xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
            <div className="mt-2 grid gap-3 lg:grid-cols-2">
              <article className="rounded-xl border border-slate-100 bg-slate-50 px-3 py-2.5 text-xs text-slate-600">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Trend Summary (last 4 hr)
                  </p>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] uppercase tracking-[0.2em] text-rose-600">Live data</span>
                    {criticalAlerts.length > 0 ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-rose-50 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-rose-600">
                        <AlertTriangle className="h-3.5 w-3.5" />
                        {criticalAlerts.length} Alert{criticalAlerts.length > 1 ? "s" : ""}
                      </span>
                    ) : null}
                  </div>
                </div>
                {criticalAlerts.length > 0 ? (
                  <div className="mt-3 flex flex-wrap gap-2">
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
                {trendData.length > 0 ? (
                  <div className="mt-3 grid grid-cols-2 gap-4">
                    {metricCharts.map((metric) => (
                      <article
                        key={metric.id}
                        className={cn(
                          "chart-card relative flex cursor-pointer flex-col overflow-hidden p-3 text-slate-900 transition hover:-translate-y-0.5",
                          !metric.alert && "border-slate-200/70 shadow-[0_14px_26px_rgba(15,23,42,0.08)]"
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
                          className="chart-alert mt-2.5 border border-slate-200 bg-white p-2"
                          data-alert={metric.alert?.direction ?? undefined}
                        >
                          <TrendLineChart
                            dataPoints={metric.data}
                            lines={metric.lines}
                            height={metric.height ?? 120}
                            suggestedYRange={metric.yRange}
                            valueFormatter={(value, datasetLabel) => metric.valueFormatter(value, datasetLabel)}
                            labelFormatter={(label) => `Time · ${label}`}
                            highlightDirection={metric.alert?.direction}
                          />
                        </div>
                        <div className="mt-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-left text-slate-900">
                          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
                            {metric.latestLabel}
                          </p>
                          <p className="text-lg font-semibold text-slate-900">{metric.latestValue}</p>
                          {metric.alert ? (
                            <p className="alert-delta-chip alert-delta-chip--rise mt-2">{metric.alert.label}</p>
                          ) : null}
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="mt-3 text-xs text-slate-400">No trend data available.</p>
                )}
              </article>
            </div>
            <div className="mt-2 rounded-xl border border-slate-100 bg-slate-50 px-3 py-2.5">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Comorbidities</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {patient.comorbidities.map((item, index) => (
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
          )}
        </main>
      </div>
    </div>
  );
}

function InfoChip({
  label,
  value,
  className,
}: {
  label: string;
  value: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-left",
        className
      )}
    >
      <p className="text-[10px] uppercase tracking-wide text-slate-400">{label}</p>
      <div className="text-sm font-semibold text-slate-900">{value}</div>
    </div>
  );
}
