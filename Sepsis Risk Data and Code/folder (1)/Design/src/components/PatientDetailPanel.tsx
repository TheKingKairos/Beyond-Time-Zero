import type { DashboardPatient } from "@/components/DashboardTable";
import {
  CONDITION_LABELS,
  getConditionBand,
  SEPSIS_SCORE_TONES,
} from "@/components/PatientTable";

interface ChecklistItem {
  title: string;
  status: string;
  pill: string;
  cardTone: string;
  meta: string;
  description: string;
}

interface TimelineEvent {
  time: string;
  label: string;
  tone: string;
}

interface PatientDetailPanelProps {
  patient: DashboardPatient | null;
  checkpoints: ChecklistItem[];
  incidentTimeline: TimelineEvent[];
}

export function PatientDetailPanel({
  patient,
  checkpoints,
  incidentTimeline,
}: PatientDetailPanelProps) {
  const conditionBand = patient ? getConditionBand(patient.sepsisScore) : null;

  return (
    <aside className="space-y-6">
      <article className="rounded-3xl border border-white/10 bg-white/[0.08] p-6 backdrop-blur">
        <header className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-400">
              Focused Patient
            </p>
            <h2 className="mt-3 text-2xl font-semibold text-white">
              {patient ? patient.patientId : "Select a patient"}
            </h2>
            <p className="text-sm text-slate-300">
              {patient ? patient.location : "Choose a patient from the rankings to view details."}
            </p>
          </div>
          {patient ? (
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-right">
              <p className="text-[11px] uppercase tracking-wide text-slate-400">Condition Score</p>
              <div className="mt-1 flex justify-end">
                <span
                  className="inline-flex min-w-[120px] items-center justify-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide"
                  style={conditionBand ? SEPSIS_SCORE_TONES[conditionBand] : undefined}
                >
                  {conditionBand ? CONDITION_LABELS[conditionBand] : "--"}
                </span>
              </div>
            </div>
          ) : null}
        </header>

        {patient ? (
          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-400">Time Since Last Vital</p>
              <p className="mt-2 font-mono text-xl text-white">
                {patient.lastVitalTime} min ago
              </p>
              <p className="text-xs text-slate-400">
                Escalate if over 45 minutes between bedside checks.
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <div className="grid grid-cols-2 gap-3 text-center">
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-slate-400">Temperature</p>
                  <p className="mt-2 font-mono text-xl text-white">
                    {patient.temp.toFixed(1)} °F
                  </p>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-slate-400">Heart Rate</p>
                  <p className="mt-2 font-mono text-xl text-white">{patient.hr} bpm</p>
                </div>
              </div>
              <p className="mt-3 text-xs text-slate-400">
                Monitor fever trajectory alongside lactate redraw.
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-400">Bundle Status</p>
              <p className="mt-2 font-mono text-xl text-white">
                {checkpoints.filter((item) => item.status !== "Scheduled").length}/
                {checkpoints.length} actions
              </p>
              <p className="text-xs text-slate-400">Use checklist below to finish open tasks.</p>
            </div>
          </div>
        ) : null}
      </article>

      <article className="rounded-3xl border border-white/10 bg-white/[0.08] p-6 backdrop-blur">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
          Response Timeline
        </h3>
        <div className="mt-5 space-y-4">
          {incidentTimeline.map((event) => (
            <div key={event.time} className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/10 font-mono text-sm text-slate-200">
                {event.time}
              </div>
              <p className={`pt-2 text-sm font-medium ${event.tone}`}>{event.label}</p>
            </div>
          ))}
        </div>
      </article>
    </aside>
  );
}
