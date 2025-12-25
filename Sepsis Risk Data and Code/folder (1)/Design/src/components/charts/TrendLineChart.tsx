import { useMemo } from "react";

import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
  type Plugin,
  type ScriptableContext,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Filler, Legend);

export interface TrendLineDefinition {
  key: string;
  name: string;
  stroke: string;
  mutedStroke?: string;
  gradientFrom: string;
  gradientTo: string;
  showArea?: boolean;
  strokeDasharray?: string;
}

interface TrendLineChartProps<T extends Record<string, unknown>> {
  dataPoints: T[];
  lines: TrendLineDefinition[];
  height?: number;
  suggestedYRange?: { min: number; max: number };
  valueFormatter?: (value: number, datasetLabel?: string) => string;
  labelFormatter?: (label: string) => string;
  highlightDirection?: "rise" | "drop";
  showPointLabels?: boolean;
  pointLabelFormatter?: (value: number, datasetLabel?: string) => string;
  pointLabelDatasetIndex?: number;
}

function withAlpha(color: string, alpha: number) {
  if (!color.startsWith("#") || color.length !== 7) {
    return color;
  }
  const r = Number.parseInt(color.slice(1, 3), 16);
  const g = Number.parseInt(color.slice(3, 5), 16);
  const b = Number.parseInt(color.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function TrendLineChart<T extends Record<string, unknown>>({
  dataPoints,
  lines,
  height = 130,
  suggestedYRange,
  valueFormatter,
  labelFormatter = (label) => label,
  highlightDirection,
  showPointLabels,
  pointLabelFormatter,
  pointLabelDatasetIndex = 0,
}: TrendLineChartProps<T>) {
  const labels = useMemo(() => dataPoints.map((point) => String(point.time ?? "")), [dataPoints]);

  const chartData = useMemo<ChartData<"line">>(() => {
    return {
      labels,
      datasets: lines.map((line) => {
        const dashValues = line.strokeDasharray?.split(" ").map((segment) => Number.parseFloat(segment)) ?? undefined;

        const baseStroke = line.stroke;
        const mutedStroke = line.mutedStroke ?? withAlpha(baseStroke, 0.35);

        return {
          label: line.name,
          data: dataPoints.map((point) => {
            const rawValue = point[line.key];
            return typeof rawValue === "number" ? rawValue : null;
          }),
          borderColor: mutedStroke,
          backgroundColor: line.showArea
            ? (context: ScriptableContext<"line">) => {
                const { chart } = context;
                const { ctx, chartArea } = chart;
                if (!chartArea) {
                  return line.gradientFrom;
                }
                const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                gradient.addColorStop(0, line.gradientFrom);
                gradient.addColorStop(1, line.gradientTo);
                return gradient;
              }
            : line.stroke,
          fill: Boolean(line.showArea),
          borderWidth: 2.4,
          borderDash: dashValues,
          pointRadius: (ctx) => {
            if (
              highlightDirection &&
              (ctx.dataIndex === dataPoints.length - 1 || ctx.dataIndex === dataPoints.length - 2)
            ) {
              return 4.8;
            }
            return 2.2;
          },
          pointHoverRadius: 5.2,
          pointBorderWidth: (ctx) =>
            highlightDirection &&
            (ctx.dataIndex === dataPoints.length - 1 || ctx.dataIndex === dataPoints.length - 2)
              ? 2
              : 1.2,
          pointBackgroundColor: (ctx) => {
            if (
              highlightDirection &&
              (ctx.dataIndex === dataPoints.length - 1 || ctx.dataIndex === dataPoints.length - 2)
            ) {
              return baseStroke;
            }
            return mutedStroke;
          },
          pointBorderColor: (ctx) =>
            highlightDirection &&
            (ctx.dataIndex === dataPoints.length - 1 || ctx.dataIndex === dataPoints.length - 2)
              ? withAlpha(baseStroke, 0.2)
              : "#e2e8f0",
          tension: 0,
          segment: {
            borderColor: (ctx) => {
              if (
                highlightDirection &&
                ctx.p1DataIndex === dataPoints.length - 1 &&
                ctx.p0DataIndex === dataPoints.length - 2
              ) {
                return baseStroke;
              }
              return mutedStroke;
            },
            borderWidth: (ctx) => {
              if (
                highlightDirection &&
                ctx.p1DataIndex === dataPoints.length - 1 &&
                ctx.p0DataIndex === dataPoints.length - 2
              ) {
                return 5;
              }
              return 2.4;
            },
            backgroundColor: (ctx) => {
              if (
                highlightDirection &&
                ctx.p1DataIndex === dataPoints.length - 1 &&
                ctx.p0DataIndex === dataPoints.length - 2
              ) {
                return withAlpha(baseStroke, 0.15);
              }
              return undefined;
            },
          },
        };
      }),
    };
  }, [dataPoints, labels, lines]);

  const chartOptions = useMemo<ChartOptions<"line">>(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: "index",
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: "#020617",
          borderColor: "#1f2937",
          borderWidth: 1,
          titleColor: "#e2e8f0",
          bodyColor: "#cbd5f5",
          padding: 10,
          callbacks: {
            label: (context) => {
              const datasetLabel = context.dataset.label ?? "";
              const value =
                typeof context.parsed.y === "number"
                  ? context.parsed.y
                  : typeof context.parsed === "number"
                    ? context.parsed
                    : 0;
              const formatted = valueFormatter ? valueFormatter(value, datasetLabel) : `${datasetLabel}: ${value}`;
              return formatted;
            },
            title: (tooltipItems) => {
              const label = tooltipItems[0]?.label ?? "";
              return labelFormatter(label);
            },
          },
        },
      },
      scales: {
        x: {
          border: { display: false },
          grid: {
            color: "rgba(15, 23, 42, 0.35)",
            drawTicks: false,
          },
          ticks: {
            color: "#94a3b8",
            font: {
              size: 10,
            },
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 4,
          },
        },
        y: {
          border: { display: false },
          grid: {
            color: "rgba(15, 23, 42, 0.25)",
            drawTicks: false,
          },
          ticks: {
            color: "#94a3b8",
            font: {
              size: 10,
            },
            maxTicksLimit: 5,
          },
          suggestedMin: suggestedYRange?.min,
          suggestedMax: suggestedYRange?.max,
        },
      },
    };
  }, [labelFormatter, suggestedYRange?.max, suggestedYRange?.min, valueFormatter]);

  const pointValueLabelPlugin = useMemo<Plugin<"line"> | null>(() => {
    if (!showPointLabels) {
      return null;
    }

    return {
      id: "point-value-labels",
      afterDatasetsDraw: (chart) => {
        const dataset = chart.data.datasets?.[pointLabelDatasetIndex];
        if (!dataset) {
          return;
        }

        const meta = chart.getDatasetMeta(pointLabelDatasetIndex);
        const formatter = pointLabelFormatter ?? valueFormatter ?? ((value: number) => `${value}`);

        meta.data.forEach((element, index) => {
          const rawValue = Array.isArray(dataset.data) ? dataset.data[index] : null;
          if (typeof rawValue !== "number" || !element) {
            return;
          }

          const { ctx } = chart;
          const { x, y } = element.tooltipPosition();
          const label = formatter(rawValue, dataset.label as string | undefined);

          ctx.save();
          ctx.font = "600 11px 'Inter', 'Helvetica Neue', sans-serif";
          ctx.fillStyle = "#0f172a";
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(label, x, y + 10);
          ctx.restore();
        });
      },
    } as Plugin<"line">;
  }, [pointLabelDatasetIndex, pointLabelFormatter, showPointLabels, valueFormatter]);

  return (
    <div style={{ height }}>
      <Line
        data={chartData}
        options={chartOptions}
        plugins={pointValueLabelPlugin ? [pointValueLabelPlugin] : undefined}
      />
    </div>
  );
}
