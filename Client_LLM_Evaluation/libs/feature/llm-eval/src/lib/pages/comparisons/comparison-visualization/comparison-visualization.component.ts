import {
  Component,
  Input,
  OnChanges,
  SimpleChanges,
  ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  MetricDifference,
  VisualizationData,
} from '@ngtx-apps/data-access/models';

@Component({
  selector: 'app-comparison-visualization',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './comparison-visualization.component.html',
  styleUrls: ['./comparison-visualization.component.scss'],
})
export class ComparisonVisualizationComponent implements OnChanges {
  @Input() visualizationData: VisualizationData | null = null;
  @Input() visualizationType: 'radar' | 'bar' | 'line' = 'radar';
  @Input() metricDifferences: MetricDifference[] = [];

  // SVG dimensions and configuration
  svgWidth = 700;
  svgHeight = 500;
  margin = { top: 50, right: 100, bottom: 80, left: 60 }; // Increased bottom margin for rotated labels

  // Derived dimensions
  get innerWidth(): number {
    return this.svgWidth - this.margin.left - this.margin.right;
  }

  get innerHeight(): number {
    return this.svgHeight - this.margin.top - this.margin.bottom;
  }

  // Chart data
  chartData: any = null;

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnChanges(changes: SimpleChanges): void {
    // Process data when inputs change
    if (
      (changes['visualizationData'] && this.visualizationData) ||
      (changes['metricDifferences'] && this.metricDifferences.length > 0) ||
      changes['visualizationType']
    ) {
      console.log('Visualization component inputs changed:', {
        visualizationType: this.visualizationType,
        visualizationData: this.visualizationData,
        metricDifferences: this.metricDifferences,
      });
      this.processData();
    }
  }

  /**
   * Process the input data based on visualization type
   */
  processData(): void {
    if (this.visualizationData) {
      // If visualization data is provided directly, use it
      this.chartData = this.visualizationData;
      console.log('Using provided visualization data:', this.chartData);
    } else if (this.metricDifferences.length > 0) {
      // Otherwise, create visualization data from metric differences
      this.chartData = this.createChartDataFromMetricDifferences();
      console.log(
        'Created chart data from metric differences:',
        this.chartData
      );
    }
    this.cdr.markForCheck();
  }

  /**
   * Create chart data from metric differences
   */
  private createChartDataFromMetricDifferences(): VisualizationData {
    // Use either name or metric_name, falling back to index if neither exists
    const labels = this.metricDifferences.map(
      (m, index) => m.name || m.metric_name || `Metric ${index + 1}`
    );

    // Create datasets based on visualization type
    let datasets;

    if (
      this.visualizationType === 'radar' ||
      this.visualizationType === 'line'
    ) {
      datasets = [
        {
          label: 'Evaluation A',
          data: this.metricDifferences.map((m) => m.evaluation_a_value),
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          borderColor: 'rgba(54, 162, 235, 1)',
          fill: this.visualizationType === 'radar',
        },
        {
          label: 'Evaluation B',
          data: this.metricDifferences.map((m) => m.evaluation_b_value),
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          borderColor: 'rgba(255, 99, 132, 1)',
          fill: this.visualizationType === 'radar',
        },
      ];
    } else {
      // bar chart
      datasets = [
        {
          label: 'Evaluation A',
          data: this.metricDifferences.map((m) => m.evaluation_a_value),
          backgroundColor: 'rgba(54, 162, 235, 0.7)',
        },
        {
          label: 'Evaluation B',
          data: this.metricDifferences.map((m) => m.evaluation_b_value),
          backgroundColor: 'rgba(255, 99, 132, 0.7)',
        },
        {
          label: 'Difference (%)',
          data: this.metricDifferences.map(
            (m) =>
              (m.percentage_change !== undefined
                ? m.percentage_change
                : m.percentage_difference) || 0
          ),
          backgroundColor: 'rgba(75, 192, 192, 0.7)',
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
   * Get path for radar chart
   */
  getRadarPath(dataIndex: number): string {
    if (
      !this.chartData ||
      !this.chartData.datasets ||
      this.chartData.datasets.length <= dataIndex
    ) {
      return '';
    }

    const dataset = this.chartData.datasets[dataIndex];
    const dataPoints = dataset.data;
    const labels = this.chartData.labels;

    if (!dataPoints || !labels || dataPoints.length === 0) {
      return '';
    }

    // Calculate the maximum value for scaling
    const maxValue = Math.max(
      ...this.chartData.datasets.flatMap((ds: any) =>
        ds.data.filter((val: any) => !isNaN(val))
      )
    );

    // If all values are NaN, return empty path
    if (!isFinite(maxValue)) {
      return '';
    }

    // Calculate radius and center
    const radius = Math.min(this.innerWidth, this.innerHeight) / 2;
    const centerX = this.margin.left + this.innerWidth / 2;
    const centerY = this.margin.top + this.innerHeight / 2;

    // Calculate the angles for each metric (equally spaced)
    const angleStep = (2 * Math.PI) / labels.length;

    // Create path
    let path = '';
    for (let i = 0; i < dataPoints.length; i++) {
      // Skip if value is NaN
      if (isNaN(dataPoints[i])) {
        continue;
      }

      // Scale the data point value to the radius
      const scaledValue = (dataPoints[i] / maxValue) * radius;

      // Calculate the x and y coordinates for this point
      const angle = i * angleStep - Math.PI / 2; // Start from top (minus pi/2)
      const x = centerX + scaledValue * Math.cos(angle);
      const y = centerY + scaledValue * Math.sin(angle);

      // Add to path
      if (path === '') {
        path += `M ${x} ${y}`;
      } else {
        path += ` L ${x} ${y}`;
      }
    }

    // Close the path if we have at least one point
    if (path !== '') {
      path += ' Z';
    }

    return path;
  }

  /**
   * Get radar axis lines
   */
  getRadarAxisLines(): { x1: number; y1: number; x2: number; y2: number }[] {
    if (!this.chartData || !this.chartData.labels) {
      return [];
    }

    const labels = this.chartData.labels;
    const radius = Math.min(this.innerWidth, this.innerHeight) / 2;
    const centerX = this.margin.left + this.innerWidth / 2;
    const centerY = this.margin.top + this.innerHeight / 2;

    const angleStep = (2 * Math.PI) / labels.length;

    return labels.map((_: any, i: number) => {
      const angle = i * angleStep - Math.PI / 2; // Start from top
      return {
        x1: centerX,
        y1: centerY,
        x2: centerX + radius * Math.cos(angle),
        y2: centerY + radius * Math.sin(angle),
      };
    });
  }

  /**
   * Get radar circular grid lines
   */
  getRadarGridCircles(): { cx: number; cy: number; r: number }[] {
    if (!this.chartData) {
      return [];
    }

    const radius = Math.min(this.innerWidth, this.innerHeight) / 2;
    const centerX = this.margin.left + this.innerWidth / 2;
    const centerY = this.margin.top + this.innerHeight / 2;

    // Create 5 concentric circles for the grid
    return [0.2, 0.4, 0.6, 0.8, 1].map((factor) => ({
      cx: centerX,
      cy: centerY,
      r: radius * factor,
    }));
  }

  /**
   * Get radar labels positions
   */
  getRadarLabelPositions(): { x: number; y: number; label: string }[] {
    if (!this.chartData || !this.chartData.labels) {
      return [];
    }

    const labels = this.chartData.labels;
    const radius = Math.min(this.innerWidth, this.innerHeight) / 2 + 20; // Add padding
    const centerX = this.margin.left + this.innerWidth / 2;
    const centerY = this.margin.top + this.innerHeight / 2;

    const angleStep = (2 * Math.PI) / labels.length;

    return labels.map((label: any, i: number) => {
      const angle = i * angleStep - Math.PI / 2; // Start from top
      return {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        label:
          typeof label === 'string'
            ? label.length > 15
              ? label.substring(0, 12) + '...'
              : label
            : 'Label ' + (i + 1),
      };
    });
  }

  /**
   * Get bar chart rectangles
   */
  getBarChartRects(): {
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
    label: string;
    value: number;
  }[] {
    if (!this.chartData || !this.chartData.datasets || !this.chartData.labels) {
      return [];
    }

    const labels = this.chartData.labels;
    const datasets = this.chartData.datasets;

    // Use only the first two datasets for the bar chart
    const barDatasets = datasets.slice(0, 2);

    // Calculate bar width and spacing
    const barGroupWidth = this.innerWidth / (labels.length || 1);
    const barWidth = Math.min(barGroupWidth * 0.35, 40); // Adjust as needed and cap max width
    const barSpacing = barGroupWidth * 0.05;

    // Calculate the maximum value for scaling
    const maxValue = Math.max(
      ...barDatasets.flatMap((ds: any) =>
        ds.data.filter((val: any) => !isNaN(val))
      )
    );

    // If all values are NaN, return empty array
    if (!isFinite(maxValue) || maxValue === 0) {
      return [];
    }

    const bars: {
      x: number;
      y: number;
      width: number;
      height: number;
      color: string;
      label: string;
      value: number;
    }[] = [];

    barDatasets.forEach((dataset: any, datasetIndex: number) => {
      dataset.data.forEach((value: number, index: number) => {
        // Skip if value is undefined, null, or NaN
        if (value === undefined || value === null || isNaN(value)) {
          return;
        }

        // Calculate bar height based on value
        const barHeight = (value / maxValue) * this.innerHeight;

        // Calculate bar position
        const x =
          this.margin.left +
          index * barGroupWidth +
          datasetIndex * (barWidth + barSpacing) +
          barGroupWidth * 0.1; // Add some padding on the left
        const y = this.margin.top + this.innerHeight - barHeight;

        bars.push({
          x,
          y,
          width: barWidth,
          height: barHeight || 1, // Ensure a minimum height for visibility
          color: dataset.backgroundColor,
          label: labels[index] || `Label ${index}`,
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
      !this.chartData ||
      !this.chartData.datasets ||
      this.chartData.datasets.length < 3
    ) {
      return { points: '', color: '' };
    }

    const labels = this.chartData.labels;
    const diffDataset = this.chartData.datasets[2]; // Difference dataset

    // Filter out NaN values
    const validDiffData = diffDataset.data.filter((val: any) => !isNaN(val));

    // If no valid data, return empty line
    if (validDiffData.length === 0) {
      return { points: '', color: '' };
    }

    // Calculate the maximum absolute difference value for scaling
    const maxAbsDiff = Math.max(
      ...validDiffData.map((val: number) => Math.abs(val))
    );

    // If maxAbsDiff is 0 or not finite, return empty line
    if (maxAbsDiff === 0 || !isFinite(maxAbsDiff)) {
      return { points: '', color: '' };
    }

    // Calculate bar group width
    const barGroupWidth = this.innerWidth / (labels.length || 1);

    // Calculate the center points of each bar group
    const points = diffDataset.data
      .map((value: number, index: number) => {
        // Skip if value is undefined, null, or NaN
        if (value === undefined || value === null || isNaN(value)) {
          return null;
        }

        // Calculate the scaled value (map from -maxAbsDiff..maxAbsDiff to -halfHeight..halfHeight)
        const halfHeight = this.innerHeight / 3; // Use 1/3 of height for difference line
        const scaledValue = (value / maxAbsDiff) * halfHeight;

        // Calculate the center of the bar group
        const x = this.margin.left + index * barGroupWidth + barGroupWidth / 2;
        const y = this.margin.top + this.innerHeight / 2 - scaledValue; // Center line

        return `${x},${y}`;
      })
      .filter((p: any) => p !== null)
      .join(' ');

    return {
      points,
      color: diffDataset.backgroundColor,
    };
  }

  /**
   * Get line chart paths
   */
  getLineChartPath(datasetIndex: number): string {
    if (
      !this.chartData ||
      !this.chartData.datasets ||
      this.chartData.datasets.length <= datasetIndex
    ) {
      return '';
    }

    const dataset = this.chartData.datasets[datasetIndex];
    const dataPoints = dataset.data;
    const labels = this.chartData.labels;

    if (!dataPoints || !labels || dataPoints.length === 0) {
      return '';
    }

    // Filter out NaN values for calculating max
    const validDataPoints = this.chartData.datasets.flatMap((ds: any) =>
      ds.data.filter((val: any) => !isNaN(val))
    );

    // If no valid data points, return empty path
    if (validDataPoints.length === 0) {
      return '';
    }

    // Calculate the maximum value for scaling
    const maxValue = Math.max(...validDataPoints);

    // If maxValue is 0 or not finite, return empty path
    if (maxValue === 0 || !isFinite(maxValue)) {
      return '';
    }

    // Calculate x and y scales
    const xStep = this.innerWidth / Math.max(1, labels.length - 1);

    let path = '';
    let isFirstValidPoint = true;

    dataPoints.forEach((value: number, index: number) => {
      // Skip if value is undefined, null, or NaN
      if (value === undefined || value === null || isNaN(value)) {
        return;
      }

      // Calculate x and y coordinates
      const x = this.margin.left + index * xStep;
      const y =
        this.margin.top +
        this.innerHeight -
        (value / maxValue) * this.innerHeight;

      // Add to path
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
   * Get data points for the line chart
   */
  getDataPoints(datasetIndex: number): { x: number; y: number }[] {
    if (
      !this.chartData ||
      !this.chartData.datasets ||
      this.chartData.datasets.length <= datasetIndex ||
      this.visualizationType !== 'line'
    ) {
      return [];
    }

    const dataset = this.chartData.datasets[datasetIndex];
    const dataPoints = dataset.data;
    const labels = this.chartData.labels;

    if (!dataPoints || !labels || dataPoints.length === 0) {
      return [];
    }

    // Filter out NaN values for calculating max
    const validDataPoints = this.chartData.datasets.flatMap((ds: any) =>
      ds.data.filter((val: any) => !isNaN(val))
    );

    // If no valid data points, return empty array
    if (validDataPoints.length === 0) {
      return [];
    }

    // Calculate the maximum value for scaling
    const maxValue = Math.max(...validDataPoints);

    // If maxValue is not valid, return empty array
    if (maxValue === 0 || !isFinite(maxValue)) {
      return [];
    }

    // Calculate x and y scales
    const xStep = this.innerWidth / Math.max(1, labels.length - 1);

    return dataPoints
      .map((value: number, index: number) => {
        // Skip if value is undefined, null, or NaN
        if (value === undefined || value === null || isNaN(value)) {
          return null;
        }

        // Calculate x and y coordinates
        const x = this.margin.left + index * xStep;
        const y =
          this.margin.top +
          this.innerHeight -
          (value / maxValue) * this.innerHeight;

        return { x, y };
      })
      .filter((point: any) => point !== null);
  }

  /**
   * Get y-axis ticks for line and bar charts
   */
  getYAxisTicks(): {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    label: string;
  }[] {
    // Generate 5 evenly spaced ticks
    const ticks = [0, 0.25, 0.5, 0.75, 1];

    return ticks.map((tick) => {
      const y = this.margin.top + this.innerHeight - tick * this.innerHeight;
      return {
        x1: this.margin.left - 5,
        y1: y,
        x2: this.margin.left,
        y2: y,
        label: (tick * 100).toFixed(0) + '%',
      };
    });
  }

  /**
   * Get x-axis labels for line and bar charts
   */
  getXAxisLabels(): { x: number; y: number; label: string }[] {
    if (!this.chartData || !this.chartData.labels) {
      return [];
    }

    const labels = this.chartData.labels;

    // Ensure we don't have too many labels that would cause cluttering
    const maxLabelsToShow = Math.min(
      labels.length,
      Math.floor(this.innerWidth / 80)
    );
    const skipFactor = Math.ceil(labels.length / maxLabelsToShow);

    const filteredLabels = labels.filter(
      (_: any, i: any) => i % skipFactor === 0 || i === labels.length - 1
    );

    if (this.visualizationType === 'bar') {
      // For bar chart, labels go in the center of each bar group
      const barGroupWidth = this.innerWidth / labels.length;

      return labels.map((label: string, index: number) => {
        // Truncate long labels
        const displayLabel =
          typeof label === 'string'
            ? label.length > 15
              ? label.substring(0, 12) + '...'
              : label
            : `Label ${index + 1}`;

        const x = this.margin.left + index * barGroupWidth + barGroupWidth / 2;
        const y = this.margin.top + this.innerHeight + 10;
        return { x, y, label: displayLabel };
      });
    } else {
      // line chart
      // For line chart, labels go at each data point, with filtering for large datasets
      const xStep = this.innerWidth / Math.max(1, labels.length - 1);

      return filteredLabels.map((label: string, i: number) => {
        // Get the actual index in the original labels array
        const originalIndex = labels.indexOf(label);

        // Truncate long labels
        const displayLabel =
          typeof label === 'string'
            ? label.length > 15
              ? label.substring(0, 12) + '...'
              : label
            : `Label ${originalIndex + 1}`;

        const x = this.margin.left + originalIndex * xStep;
        const y = this.margin.top + this.innerHeight + 10;
        return { x, y, label: displayLabel };
      });
    }
  }
}
