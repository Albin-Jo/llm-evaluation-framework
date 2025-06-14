import {
  Component,
  Input,
  OnChanges,
  SimpleChanges,
  ChangeDetectorRef,
  ElementRef,
  ViewChild,
  AfterViewInit,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  MetricDifference,
  VisualizationData,
} from '@ngtx-apps/data-access/models';

interface ChartPoint {
  x: number;
  y: number;
  value: number;
  label: string;
}

interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string;
  borderColor?: string;
  fill?: boolean;
}

@Component({
  selector: 'app-comparison-visualization',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './comparison-visualization.component.html',
  styleUrls: ['./comparison-visualization.component.scss'],
})
export class ComparisonVisualizationComponent
  implements OnChanges, AfterViewInit
{
  @Input() visualizationData: VisualizationData | null = null;
  @Input() visualizationType: 'radar' | 'bar' | 'line' = 'radar';
  @Input() metricDifferences: MetricDifference[] = [];

  @ViewChild('chartContainer', { static: false }) chartContainer!: ElementRef;

  // SVG dimensions and configuration
  svgWidth = 800;
  svgHeight = 600;
  margin = { top: 80, right: 120, bottom: 120, left: 80 };

  // Derived dimensions
  get innerWidth(): number {
    return this.svgWidth - this.margin.left - this.margin.right;
  }

  get innerHeight(): number {
    return this.svgHeight - this.margin.top - this.margin.bottom;
  }

  // Chart data
  chartData: any = null;
  processedData: any = null;

  // Color schemes
  private readonly colorSchemes = {
    primary: 'rgba(54, 162, 235, 0.8)',
    primaryLight: 'rgba(54, 162, 235, 0.2)',
    secondary: 'rgba(255, 99, 132, 0.8)',
    secondaryLight: 'rgba(255, 99, 132, 0.2)',
    tertiary: 'rgba(75, 192, 192, 0.8)',
    tertiaryLight: 'rgba(75, 192, 192, 0.2)',
    grid: '#e0e0e0',
    axis: '#757575',
    text: '#212121',
  };

  constructor(private cdr: ChangeDetectorRef) {}

  ngAfterViewInit(): void {
    this.updateSvgDimensions();
    this.processData();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (
      (changes['visualizationData'] && this.visualizationData) ||
      (changes['metricDifferences'] && this.metricDifferences.length > 0) ||
      changes['visualizationType']
    ) {
      this.processData();
    }
  }

  /**
   * Update SVG dimensions based on container size
   */
  private updateSvgDimensions(): void {
    if (this.chartContainer?.nativeElement) {
      const containerWidth = this.chartContainer.nativeElement.offsetWidth;
      if (containerWidth > 0) {
        this.svgWidth = Math.min(containerWidth - 40, 1000);
        this.svgHeight = Math.max(400, this.svgWidth * 0.6);

        // Adjust margins for smaller screens
        if (this.svgWidth < 600) {
          this.margin = { top: 60, right: 80, bottom: 100, left: 60 };
        }
      }
    }
  }

  /**
   * Process the input data based on visualization type
   */
  processData(): void {
    if (this.visualizationData) {
      this.chartData = this.visualizationData;
    } else if (this.metricDifferences.length > 0) {
      this.chartData = this.createChartDataFromMetricDifferences();
    }

    if (this.chartData) {
      this.processedData = this.processChartData(this.chartData);
      this.cdr.markForCheck();
    }
  }

  /**
   * Create chart data from metric differences
   */
  private createChartDataFromMetricDifferences(): VisualizationData {
    const labels = this.metricDifferences.map(
      (m, index) => m.name || m.metric_name || `Metric ${index + 1}`
    );

    let datasets: ChartDataset[];

    if (
      this.visualizationType === 'radar' ||
      this.visualizationType === 'line'
    ) {
      datasets = [
        {
          label: 'Evaluation A',
          data: this.metricDifferences.map((m) => m.evaluation_a_value),
          backgroundColor: this.colorSchemes.primaryLight,
          borderColor: this.colorSchemes.primary,
          fill: this.visualizationType === 'radar',
        },
        {
          label: 'Evaluation B',
          data: this.metricDifferences.map((m) => m.evaluation_b_value),
          backgroundColor: this.colorSchemes.secondaryLight,
          borderColor: this.colorSchemes.secondary,
          fill: this.visualizationType === 'radar',
        },
      ];
    } else {
      datasets = [
        {
          label: 'Evaluation A',
          data: this.metricDifferences.map((m) => m.evaluation_a_value),
          backgroundColor: this.colorSchemes.primary,
        },
        {
          label: 'Evaluation B',
          data: this.metricDifferences.map((m) => m.evaluation_b_value),
          backgroundColor: this.colorSchemes.secondary,
        },
        {
          label: 'Difference (%)',
          data: this.metricDifferences.map((m) => m.percentage_change || 0),
          backgroundColor: this.colorSchemes.tertiary,
        },
      ];
    }

    return {
      type: this.visualizationType,
      labels,
      datasets,
    };
  }

  /**
   * Process chart data for rendering
   */
  private processChartData(data: any): any {
    if (!data || !data.labels || !data.datasets) {
      return null;
    }

    const processed = {
      labels: data.labels,
      datasets: data.datasets,
      maxValue: 0,
      minValue: 0,
    };

    // Calculate min and max values for scaling
    const allValues = data.datasets.flatMap((dataset: any) =>
      dataset.data.filter((val: any) => !isNaN(val) && isFinite(val))
    );

    if (allValues.length > 0) {
      processed.maxValue = Math.max(...allValues);
      processed.minValue = Math.min(...allValues);

      // For difference data (can be negative), adjust range
      if (this.visualizationType === 'bar' && data.datasets.length > 2) {
        const diffValues = data.datasets[2].data.filter(
          (val: any) => !isNaN(val) && isFinite(val)
        );
        if (diffValues.length > 0) {
          processed.maxValue = Math.max(
            processed.maxValue,
            Math.max(...diffValues)
          );
          processed.minValue = Math.min(
            processed.minValue,
            Math.min(...diffValues)
          );
        }
      }

      // Add some padding
      const range = processed.maxValue - processed.minValue;
      processed.maxValue += range * 0.1;
      processed.minValue -= range * 0.1;

      // Ensure minimum range for radar and line charts
      if (this.visualizationType !== 'bar') {
        processed.minValue = Math.max(0, processed.minValue);
        if (processed.maxValue < 1) {
          processed.maxValue = 1;
        }
      }
    }

    return processed;
  }

  /**
   * Get radar chart path
   */
  getRadarPath(datasetIndex: number): string {
    if (!this.processedData || !this.processedData.datasets[datasetIndex]) {
      return '';
    }

    const dataset = this.processedData.datasets[datasetIndex];
    const dataPoints = dataset.data;
    const labels = this.processedData.labels;

    if (!dataPoints || !labels || dataPoints.length === 0) {
      return '';
    }

    const radius = Math.min(this.innerWidth, this.innerHeight) / 2;
    const centerX = this.margin.left + this.innerWidth / 2;
    const centerY = this.margin.top + this.innerHeight / 2;
    const angleStep = (2 * Math.PI) / labels.length;

    let path = '';
    let hasValidPoints = false;

    for (let i = 0; i < dataPoints.length; i++) {
      if (isNaN(dataPoints[i]) || !isFinite(dataPoints[i])) continue;

      const scaledValue =
        (dataPoints[i] / this.processedData.maxValue) * radius;
      const angle = i * angleStep - Math.PI / 2;
      const x = centerX + scaledValue * Math.cos(angle);
      const y = centerY + scaledValue * Math.sin(angle);

      if (!hasValidPoints) {
        path += `M ${x} ${y}`;
        hasValidPoints = true;
      } else {
        path += ` L ${x} ${y}`;
      }
    }

    return hasValidPoints ? path + ' Z' : '';
  }

  /**
   * Get radar axis lines
   */
  getRadarAxisLines(): { x1: number; y1: number; x2: number; y2: number }[] {
    if (!this.processedData?.labels) return [];

    const labels = this.processedData.labels;
    const radius = Math.min(this.innerWidth, this.innerHeight) / 2;
    const centerX = this.margin.left + this.innerWidth / 2;
    const centerY = this.margin.top + this.innerHeight / 2;
    const angleStep = (2 * Math.PI) / labels.length;

    return labels.map((_: any, i: number) => {
      const angle = i * angleStep - Math.PI / 2;
      return {
        x1: centerX,
        y1: centerY,
        x2: centerX + radius * Math.cos(angle),
        y2: centerY + radius * Math.sin(angle),
      };
    });
  }

  /**
   * Get radar grid circles
   */
  getRadarGridCircles(): { cx: number; cy: number; r: number }[] {
    if (!this.processedData) return [];

    const radius = Math.min(this.innerWidth, this.innerHeight) / 2;
    const centerX = this.margin.left + this.innerWidth / 2;
    const centerY = this.margin.top + this.innerHeight / 2;

    return [0.2, 0.4, 0.6, 0.8, 1].map((factor) => ({
      cx: centerX,
      cy: centerY,
      r: radius * factor,
    }));
  }

  /**
   * Get radar label positions
   */
  getRadarLabelPositions(): { x: number; y: number; label: string }[] {
    if (!this.processedData?.labels) return [];

    const labels = this.processedData.labels;
    const radius = Math.min(this.innerWidth, this.innerHeight) / 2 + 30;
    const centerX = this.margin.left + this.innerWidth / 2;
    const centerY = this.margin.top + this.innerHeight / 2;
    const angleStep = (2 * Math.PI) / labels.length;

    return labels.map((label: any, i: number) => {
      const angle = i * angleStep - Math.PI / 2;
      const truncatedLabel = this.truncateLabel(label);

      return {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        label: truncatedLabel,
      };
    });
  }

  /**
   * Get bar chart rectangles
   */
  getBarChartRects(): Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
    value: number;
  }> {
    if (!this.processedData?.datasets || !this.processedData?.labels) return [];

    const labels = this.processedData.labels;
    const datasets = this.processedData.datasets.slice(0, 2); // Only first two datasets for bars

    const barGroupWidth = this.innerWidth / labels.length;
    const barWidth = Math.min(barGroupWidth * 0.35, 50);
    const barSpacing = barGroupWidth * 0.1;

    const bars: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      color: string;
      value: number;
    }> = [];

    datasets.forEach((dataset: any, datasetIndex: number) => {
      dataset.data.forEach((value: number, index: number) => {
        if (isNaN(value) || !isFinite(value)) return;

        const normalizedValue = Math.max(0, value);
        const barHeight =
          (normalizedValue / this.processedData.maxValue) * this.innerHeight;

        const x =
          this.margin.left +
          index * barGroupWidth +
          datasetIndex * (barWidth + barSpacing) +
          (barGroupWidth -
            (datasets.length * barWidth + (datasets.length - 1) * barSpacing)) /
            2;
        const y = this.margin.top + this.innerHeight - barHeight;

        bars.push({
          x,
          y,
          width: barWidth,
          height: Math.max(1, barHeight),
          color: dataset.backgroundColor || this.colorSchemes.primary,
          value,
        });
      });
    });

    return bars;
  }

  /**
   * Get difference line for bar chart
   */
  getDifferenceLine(): { points: string; color: string } {
    if (
      !this.processedData?.datasets ||
      this.processedData.datasets.length < 3
    ) {
      return { points: '', color: '' };
    }

    const diffDataset = this.processedData.datasets[2];
    const labels = this.processedData.labels;
    const barGroupWidth = this.innerWidth / labels.length;

    // Create a scale for the difference line (centered around zero)
    const maxAbsDiff = Math.max(
      ...diffDataset.data.map((val: number) => Math.abs(val))
    );
    const zeroLine = this.margin.top + this.innerHeight / 2;
    const diffScale = (this.innerHeight * 0.3) / maxAbsDiff; // Use 30% of chart height

    const points = diffDataset.data
      .map((value: number, index: number) => {
        if (isNaN(value) || !isFinite(value)) return null;

        const x = this.margin.left + index * barGroupWidth + barGroupWidth / 2;
        const y = zeroLine - value * diffScale;

        return `${x},${y}`;
      })
      .filter((p: any) => p !== null)
      .join(' ');

    return {
      points,
      color: diffDataset.backgroundColor || this.colorSchemes.tertiary,
    };
  }

  /**
   * Get line chart path
   */
  getLineChartPath(datasetIndex: number): string {
    if (!this.processedData?.datasets?.[datasetIndex]) return '';

    const dataset = this.processedData.datasets[datasetIndex];
    const dataPoints = dataset.data;
    const labels = this.processedData.labels;

    if (!dataPoints || !labels || dataPoints.length === 0) return '';

    const xStep = this.innerWidth / Math.max(1, labels.length - 1);

    let path = '';
    let isFirstValidPoint = true;

    dataPoints.forEach((value: number, index: number) => {
      if (isNaN(value) || !isFinite(value)) return;

      const x = this.margin.left + index * xStep;
      const y =
        this.margin.top +
        this.innerHeight -
        (value / this.processedData.maxValue) * this.innerHeight;

      if (isFirstValidPoint) {
        path += `M ${x} ${y}`;
        isFirstValidPoint = false;
      } else {
        path += ` L ${x} ${y}`;
      }
    });

    return path;
  }

  /**
   * Get data points for line chart
   */
  getDataPoints(datasetIndex: number): ChartPoint[] {
    if (
      !this.processedData?.datasets?.[datasetIndex] ||
      this.visualizationType !== 'line'
    ) {
      return [];
    }

    const dataset = this.processedData.datasets[datasetIndex];
    const dataPoints = dataset.data;
    const labels = this.processedData.labels;

    if (!dataPoints || !labels || dataPoints.length === 0) return [];

    const xStep = this.innerWidth / Math.max(1, labels.length - 1);

    return dataPoints
      .map((value: number, index: number) => {
        if (isNaN(value) || !isFinite(value)) return null;

        const x = this.margin.left + index * xStep;
        const y =
          this.margin.top +
          this.innerHeight -
          (value / this.processedData.maxValue) * this.innerHeight;

        return {
          x,
          y,
          value,
          label: labels[index],
        };
      })
      .filter((point: any) => point !== null) as ChartPoint[];
  }

  /**
   * Get Y-axis ticks
   */
  getYAxisTicks(): Array<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    label: string;
  }> {
    if (!this.processedData) return [];

    const tickCount = 5;
    const ticks: Array<{
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      label: string;
    }> = [];

    for (let i = 0; i <= tickCount; i++) {
      const value = (this.processedData.maxValue / tickCount) * i;
      const y =
        this.margin.top + this.innerHeight - (i / tickCount) * this.innerHeight;

      ticks.push({
        x1: this.margin.left - 5,
        y1: y,
        x2: this.margin.left,
        y2: y,
        label: value.toFixed(2),
      });
    }

    return ticks;
  }

  /**
   * Get X-axis labels
   */
  getXAxisLabels(): Array<{ x: number; y: number; label: string }> {
    if (!this.processedData?.labels) return [];

    const labels = this.processedData.labels;

    if (this.visualizationType === 'bar') {
      const barGroupWidth = this.innerWidth / labels.length;
      return labels.map((label: string, index: number) => ({
        x: this.margin.left + index * barGroupWidth + barGroupWidth / 2,
        y: this.margin.top + this.innerHeight + 20,
        label: this.truncateLabel(label),
      }));
    } else {
      const xStep = this.innerWidth / Math.max(1, labels.length - 1);
      const maxLabels = Math.floor(this.innerWidth / 80);
      const skipFactor = Math.ceil(labels.length / maxLabels);

      return labels
        .filter(
          (_: any, i: number) => i % skipFactor === 0 || i === labels.length - 1
        )
        .map((label: string, i: number) => {
          const originalIndex = labels.indexOf(label);
          return {
            x: this.margin.left + originalIndex * xStep,
            y: this.margin.top + this.innerHeight + 20,
            label: this.truncateLabel(label),
          };
        });
    }
  }

  /**
   * Truncate long labels
   */
  private truncateLabel(label: string): string {
    if (typeof label !== 'string') return String(label);

    const maxLength = this.svgWidth < 600 ? 8 : 12;
    return label.length > maxLength
      ? label.substring(0, maxLength) + '...'
      : label;
  }

  /**
   * Get chart title
   */
  getChartTitle(): string {
    switch (this.visualizationType) {
      case 'radar':
        return 'Radar Chart Comparison';
      case 'bar':
        return 'Bar Chart Comparison';
      case 'line':
        return 'Line Chart Comparison';
      default:
        return 'Metric Comparison';
    }
  }

  /**
   * Check if data is available
   */
  hasData(): boolean {
    return !!(
      this.processedData?.datasets?.length > 0 &&
      this.processedData?.labels?.length > 0
    );
  }

  /**
   * Get SVG viewBox
   */
  getViewBox(): string {
    return `0 0 ${this.svgWidth} ${this.svgHeight}`;
  }
}
