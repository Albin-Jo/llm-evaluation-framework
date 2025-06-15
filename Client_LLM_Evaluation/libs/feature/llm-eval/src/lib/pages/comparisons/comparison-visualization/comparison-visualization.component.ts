import {
  Component,
  Input,
  OnInit,
  OnChanges,
  SimpleChanges,
  ViewChild,
  ElementRef,
  ChangeDetectorRef,
  OnDestroy,
  NO_ERRORS_SCHEMA,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  MetricDifference,
  ApiVisualizationData,
  RadarVisualizationData,
  BarVisualizationData,
  LineVisualizationData,
  ChartVisualizationData,
  ChartDataset,
} from '@ngtx-apps/data-access/models';

// Global Chart.js declaration
declare const Chart: any;

@Component({
  selector: 'app-comparison-visualization',
  standalone: true,
  imports: [CommonModule],
  schemas: [NO_ERRORS_SCHEMA],
  template: `
    <div class="visualization-container">
      <!-- Loading State -->
      <div *ngIf="isLoading" class="chart-loading">
        <div class="spinner"></div>
        <p>Loading {{ visualizationType }} chart...</p>
      </div>

      <!-- Error State -->
      <div *ngIf="error && !showFallback" class="chart-error">
        <i class="icon icon-alert-circle"></i>
        <p>{{ error }}</p>
        <div class="error-actions">
          <button class="btn btn-outline btn-sm" (click)="retryChart()">
            <i class="icon icon-refresh"></i>
            Retry Chart
          </button>
          <button class="btn btn-outline btn-sm" (click)="showTableFallback()">
            <i class="icon icon-table"></i>
            Show Table
          </button>
        </div>
      </div>

      <!-- Chart Canvas -->
      <div
        class="chart-wrapper"
        [style.display]="isChartReady && !error ? 'block' : 'none'"
      >
        <canvas
          #chartCanvas
          class="chart-canvas"
          width="400"
          height="300"
          [attr.aria-label]="getChartAriaLabel()"
          role="img"
        >
        </canvas>
      </div>

      <!-- Fallback Table View -->
      <div *ngIf="showFallback" class="fallback-view">
        <div class="fallback-header">
          <h4>
            <i class="icon icon-table"></i>
            Metrics Comparison Table
          </h4>
          <button
            class="btn btn-outline btn-sm"
            (click)="retryChart()"
            *ngIf="error"
          >
            <i class="icon icon-bar-chart"></i>
            Try Chart Again
          </button>
        </div>

        <div class="table-container" *ngIf="tableData.length > 0">
          <table class="fallback-table">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col" *ngFor="let col of tableColumns">{{ col }}</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let row of tableData; trackBy: trackByMetricName">
                <td class="metric-name">{{ formatMetricName(row.metric) }}</td>
                <td *ngFor="let value of row.values" class="metric-value">
                  {{ formatNumber(value) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div *ngIf="tableData.length === 0" class="no-data">
          <i class="icon icon-info"></i>
          <p>No data available for {{ visualizationType }} visualization.</p>
        </div>
      </div>
    </div>
  `,
  styles: [
    `
      .visualization-container {
        width: 100%;
        height: 100%;
        position: relative;
        min-height: 350px;
      }

      .chart-loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 350px;
        color: #6b7280;
        gap: 12px;
      }

      .spinner {
        width: 24px;
        height: 24px;
        border: 2px solid #e5e7eb;
        border-radius: 50%;
        border-top-color: #3b82f6;
        animation: spin 1s linear infinite;
      }

      @keyframes spin {
        to {
          transform: rotate(360deg);
        }
      }

      .chart-error {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 350px;
        color: #ef4444;
        text-align: center;
        gap: 12px;
      }

      .chart-error .icon {
        font-size: 32px;
      }

      .chart-error p {
        color: #374151;
        margin: 0;
        max-width: 300px;
      }

      .error-actions {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: center;
      }

      .chart-wrapper {
        width: 100%;
        height: 350px;
        position: relative;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
      }

      .chart-canvas {
        width: 100% !important;
        height: 100% !important;
        max-width: 100%;
        max-height: 320px;
      }

      .fallback-view {
        padding: 16px;
      }

      .fallback-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        flex-wrap: wrap;
        gap: 12px;
      }

      .fallback-header h4 {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 16px;
        font-weight: 600;
        margin: 0;
        color: #111827;
      }

      .table-container {
        overflow-x: auto;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        max-height: 400px;
        overflow-y: auto;
      }

      .fallback-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        background: white;
      }

      .fallback-table th,
      .fallback-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
      }

      .fallback-table th {
        background: #f9fafb;
        font-weight: 600;
        color: #374151;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        position: sticky;
        top: 0;
        z-index: 1;
      }

      .fallback-table tbody tr:hover {
        background: #f9fafb;
      }

      .metric-name {
        font-weight: 500;
        color: #111827;
        min-width: 150px;
      }

      .metric-value {
        font-family: monospace;
        text-align: right;
        color: #374151;
      }

      .no-data {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 48px 32px;
        color: #6b7280;
        gap: 12px;
      }

      .no-data .icon {
        font-size: 32px;
      }

      .btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        font-size: 12px;
        color: #374151;
        cursor: pointer;
        transition: all 0.2s;
        text-decoration: none;
      }

      .btn:hover {
        background: #f9fafb;
        border-color: #3b82f6;
        color: #3b82f6;
      }

      .btn .icon {
        font-size: 12px;
      }

      @media (max-width: 768px) {
        .visualization-container {
          min-height: 300px;
        }

        .chart-wrapper {
          height: 300px;
          padding: 8px;
        }

        .chart-canvas {
          max-height: 280px;
        }

        .fallback-header {
          flex-direction: column;
          align-items: flex-start;
        }

        .fallback-table {
          font-size: 12px;
        }

        .fallback-table th,
        .fallback-table td {
          padding: 8px;
        }
      }
    `,
  ],
})
export class ComparisonVisualizationComponent
  implements OnInit, OnChanges, OnDestroy
{
  @Input() visualizationType: 'radar' | 'bar' | 'line' = 'radar';
  @Input() visualizationData: ApiVisualizationData | null = null;
  @Input() metricDifferences: MetricDifference[] = [];

  @ViewChild('chartCanvas', { static: true })
  chartCanvas!: ElementRef<HTMLCanvasElement>;

  private chart: any = null;

  isLoading: boolean = false;
  isChartReady: boolean = false;
  error: string | null = null;
  showFallback: boolean = false;

  // Table data for fallback
  tableData: Array<{ metric: string; values: number[] }> = [];
  tableColumns: string[] = [];

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.initializeChart();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (
      changes['visualizationType'] ||
      changes['visualizationData'] ||
      changes['metricDifferences']
    ) {
      this.initializeChart();
    }
  }

  ngOnDestroy(): void {
    this.destroyChart();
  }

  private async initializeChart(): Promise<void> {
    try {
      this.isLoading = true;
      this.error = null;
      this.showFallback = false;
      this.cdr.markForCheck();

      // Check if we have data
      if (!this.hasValidData()) {
        throw new Error('No data available for visualization');
      }

      // Ensure Chart.js is loaded
      await this.ensureChartJsLoaded();

      // Wait for view to be ready
      await this.waitForView();

      // Create the chart
      await this.createChart();
    } catch (error) {
      console.error('Chart initialization error:', error);
      this.handleChartError(`Chart could not be created: ${error}`);
    } finally {
      this.isLoading = false;
      this.cdr.markForCheck();
    }
  }

  private hasValidData(): boolean {
    if (this.visualizationData) {
      return this.validateApiData(this.visualizationData);
    }
    return this.metricDifferences && this.metricDifferences.length > 0;
  }

  private validateApiData(data: ApiVisualizationData): boolean {
    switch (data.type) {
      case 'radar':
        const radarData = data as RadarVisualizationData;
        return !!(
          radarData.labels &&
          radarData.labels.length > 0 &&
          radarData.series &&
          radarData.series.length > 0
        );

      case 'bar':
        const barData = data as BarVisualizationData;
        return !!(
          barData.categories &&
          barData.categories.length > 0 &&
          barData.series &&
          barData.series.length > 0
        );

      case 'line':
        const lineData = data as LineVisualizationData;
        return !!(lineData.metrics && lineData.metrics.length > 0);

      default:
        return false;
    }
  }

  private async ensureChartJsLoaded(): Promise<void> {
    if (typeof Chart !== 'undefined') {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src =
        'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js';
      script.onload = () => {
        this.registerChartComponents();
        resolve();
      };
      script.onerror = () => {
        reject(new Error('Failed to load Chart.js from CDN'));
      };
      document.head.appendChild(script);
    });
  }

  private registerChartComponents(): void {
    if (typeof Chart === 'undefined') return;

    try {
      Chart.register(
        Chart.CategoryScale,
        Chart.LinearScale,
        Chart.RadialLinearScale,
        Chart.PointElement,
        Chart.LineElement,
        Chart.BarElement,
        Chart.Title,
        Chart.Tooltip,
        Chart.Legend,
        Chart.Filler
      );
    } catch (error) {
      console.warn('Chart.js component registration failed:', error);
    }
  }

  private waitForView(): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
  }

  private async createChart(): Promise<void> {
    this.destroyChart();

    const canvas = this.chartCanvas?.nativeElement;
    if (!canvas) {
      throw new Error('Canvas element not found');
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get canvas 2D context');
    }

    // Convert API data to Chart.js format
    const chartData = this.convertToChartData();
    if (!chartData || !chartData.labels || chartData.labels.length === 0) {
      throw new Error('No valid data for visualization');
    }

    // Generate table data for fallback
    this.generateTableData(chartData);

    // Create chart configuration
    const config = this.createChartConfig(chartData);

    // Create the chart
    this.chart = new Chart(ctx, config);
    this.isChartReady = true;
    this.error = null;
  }

  private convertToChartData(): ChartVisualizationData | null {
    if (this.visualizationData) {
      return this.convertApiDataToChart(this.visualizationData);
    }

    // Fallback to creating from metric differences
    return this.createChartDataFromMetrics();
  }

  private convertApiDataToChart(
    apiData: ApiVisualizationData
  ): ChartVisualizationData | null {
    switch (apiData.type) {
      case 'radar':
        return this.convertRadarData(apiData as RadarVisualizationData);

      case 'bar':
        return this.convertBarData(apiData as BarVisualizationData);

      case 'line':
        return this.convertLineData(apiData as LineVisualizationData);

      default:
        console.warn('Unknown visualization type:', (apiData as any).type);
        return null;
    }
  }

  private convertRadarData(
    data: RadarVisualizationData
  ): ChartVisualizationData {
    const colors = [
      'rgba(59, 130, 246, 0.8)', // Blue
      'rgba(16, 185, 129, 0.8)', // Green
      'rgba(245, 158, 11, 0.8)', // Orange
      'rgba(239, 68, 68, 0.8)', // Red
      'rgba(139, 92, 246, 0.8)', // Purple
    ];

    const datasets: ChartDataset[] = data.series.map((series, index) => ({
      label: series.name,
      data: series.data,
      backgroundColor: colors[index % colors.length].replace('0.8', '0.2'),
      borderColor: colors[index % colors.length],
      borderWidth: 2,
      fill: true,
      pointRadius: 4,
      pointHoverRadius: 6,
    }));

    return {
      type: 'radar',
      labels: data.labels.map((label) => this.formatMetricName(label)),
      datasets,
    };
  }

  private convertBarData(data: BarVisualizationData): ChartVisualizationData {
    const colors = [
      'rgba(59, 130, 246, 0.8)', // Blue
      'rgba(16, 185, 129, 0.8)', // Green
      'rgba(245, 158, 11, 0.8)', // Orange
      'rgba(239, 68, 68, 0.8)', // Red
    ];

    const datasets: ChartDataset[] = data.series.map((series, index) => {
      const dataset: ChartDataset = {
        label: series.name,
        data: series.data,
        backgroundColor: colors[index % colors.length],
        borderColor: colors[index % colors.length].replace('0.8', '1'),
        borderWidth: 1,
      };

      // Handle mixed bar/line charts
      if (series.type === 'line') {
        dataset.type = 'line';
        dataset.backgroundColor = colors[index % colors.length].replace(
          '0.8',
          '0.2'
        );
        dataset.borderWidth = 2;
        dataset.fill = false;
        dataset.tension = 0.1;
        dataset.yAxisID = 'y1'; // Secondary axis for change data
      }

      return dataset;
    });

    return {
      type: 'bar',
      labels: data.categories.map((label) => this.formatMetricName(label)),
      datasets,
    };
  }

  private convertLineData(data: LineVisualizationData): ChartVisualizationData {
    const labels = data.metrics.map((m) => this.formatMetricName(m.name));

    const evaluationAData = data.metrics.map((m) => m.evaluation_a.median);
    const evaluationBData = data.metrics.map((m) => m.evaluation_b.median);

    const datasets: ChartDataset[] = [
      {
        label: data.metrics[0]?.evaluation_a.name || 'Evaluation A',
        data: evaluationAData,
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 4,
        pointHoverRadius: 6,
      },
      {
        label: data.metrics[0]?.evaluation_b.name || 'Evaluation B',
        data: evaluationBData,
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderColor: 'rgba(16, 185, 129, 1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 4,
        pointHoverRadius: 6,
      },
    ];

    return {
      type: 'line',
      labels,
      datasets,
    };
  }

  private createChartDataFromMetrics(): ChartVisualizationData | null {
    if (!this.metricDifferences || this.metricDifferences.length === 0) {
      return null;
    }

    const labels = this.metricDifferences.map((m) =>
      this.formatMetricName(m.metric_name)
    );
    const evaluationAData = this.metricDifferences.map(
      (m) => m.evaluation_a_value || 0
    );
    const evaluationBData = this.metricDifferences.map(
      (m) => m.evaluation_b_value || 0
    );

    const datasets = this.createDatasetsFromMetrics(
      evaluationAData,
      evaluationBData
    );

    return {
      type: this.visualizationType,
      labels,
      datasets,
    };
  }

  private createDatasetsFromMetrics(
    evaluationAData: number[],
    evaluationBData: number[]
  ): ChartDataset[] {
    if (this.visualizationType === 'radar') {
      return [
        {
          label: 'Evaluation A',
          data: evaluationAData,
          backgroundColor: 'rgba(59, 130, 246, 0.2)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 2,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
        {
          label: 'Evaluation B',
          data: evaluationBData,
          backgroundColor: 'rgba(16, 185, 129, 0.2)',
          borderColor: 'rgba(16, 185, 129, 1)',
          borderWidth: 2,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
      ];
    } else if (this.visualizationType === 'line') {
      return [
        {
          label: 'Evaluation A',
          data: evaluationAData,
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
        {
          label: 'Evaluation B',
          data: evaluationBData,
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          borderColor: 'rgba(16, 185, 129, 1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
      ];
    } else {
      // Bar chart
      return [
        {
          label: 'Evaluation A',
          data: evaluationAData,
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 1,
        },
        {
          label: 'Evaluation B',
          data: evaluationBData,
          backgroundColor: 'rgba(16, 185, 129, 0.8)',
          borderColor: 'rgba(16, 185, 129, 1)',
          borderWidth: 1,
        },
      ];
    }
  }

  private generateTableData(chartData: ChartVisualizationData): void {
    this.tableColumns = chartData.datasets.map((d) => d.label);
    this.tableData = chartData.labels.map((label, index) => ({
      metric: label,
      values: chartData.datasets.map((d) => d.data[index] || 0),
    }));
  }

  private createChartConfig(data: ChartVisualizationData): any {
    const baseConfig = {
      type: data.type,
      data: {
        labels: data.labels,
        datasets: data.datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top' as const,
            labels: {
              usePointStyle: true,
              padding: 16,
              font: {
                size: 12,
              },
            },
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: 'white',
            bodyColor: 'white',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            cornerRadius: 6,
            callbacks: {
              label: (context: any) => {
                const label = context.dataset.label || '';
                const value = context.parsed;
                const formattedValue =
                  typeof value === 'object'
                    ? this.formatNumber(value.y || value.r || 0)
                    : this.formatNumber(value || 0);
                return `${label}: ${formattedValue}`;
              },
            },
          },
        },
        scales: this.createScalesConfig(data),
        animation: {
          duration: 750,
          easing: 'easeInOutQuart',
        },
      },
    };

    return baseConfig;
  }

  private createScalesConfig(data: ChartVisualizationData): any {
    if (data.type === 'radar') {
      return {
        r: {
          angleLines: {
            display: true,
            color: 'rgba(0, 0, 0, 0.1)',
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)',
          },
          pointLabels: {
            font: {
              size: 11,
            },
            color: '#374151',
          },
          ticks: {
            font: {
              size: 10,
            },
            color: '#6b7280',
            backdropColor: 'transparent',
            callback: (value: number) => this.formatNumber(value),
          },
        },
      };
    }

    // Check if we need dual y-axes for bar/line combination
    const hasLineType = data.datasets.some((d) => d.type === 'line');

    const scales: any = {
      x: {
        grid: {
          display: data.type === 'line',
          color: 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          font: {
            size: 11,
          },
          color: '#374151',
          maxRotation: 45,
        },
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          font: {
            size: 11,
          },
          color: '#374151',
          callback: (value: number) => this.formatNumber(value),
        },
      },
    };

    // Add secondary y-axis for change data in bar charts
    if (hasLineType) {
      scales.y1 = {
        type: 'linear',
        display: true,
        position: 'right',
        grid: {
          drawOnChartArea: false,
        },
        ticks: {
          font: {
            size: 11,
          },
          color: '#374151',
          callback: (value: number) => `${this.formatNumber(value)}%`,
        },
      };
    }

    return scales;
  }

  private destroyChart(): void {
    if (this.chart) {
      try {
        this.chart.destroy();
      } catch (error) {
        console.warn('Error destroying chart:', error);
      }
      this.chart = null;
    }
    this.isChartReady = false;
  }

  private handleChartError(message: string): void {
    this.error = message;
    this.isChartReady = false;
    this.destroyChart();

    // Generate table data for fallback
    if (this.visualizationData) {
      this.generateTableDataFromApi();
    } else if (this.metricDifferences.length > 0) {
      this.generateTableDataFromMetrics();
    }
  }

  private generateTableDataFromApi(): void {
    if (!this.visualizationData) return;

    switch (this.visualizationData.type) {
      case 'radar':
        const radarData = this.visualizationData as RadarVisualizationData;
        this.tableColumns = radarData.series.map((s) => s.name);
        this.tableData = radarData.labels.map((label, index) => ({
          metric: this.formatMetricName(label),
          values: radarData.series.map((s) => s.data[index] || 0),
        }));
        break;

      case 'bar':
        const barData = this.visualizationData as BarVisualizationData;
        this.tableColumns = barData.series.map((s) => s.name);
        this.tableData = barData.categories.map((label, index) => ({
          metric: this.formatMetricName(label),
          values: barData.series.map((s) => s.data[index] || 0),
        }));
        break;

      case 'line':
        const lineData = this.visualizationData as LineVisualizationData;
        this.tableColumns = ['Evaluation A (Median)', 'Evaluation B (Median)'];
        this.tableData = lineData.metrics.map((metric) => ({
          metric: this.formatMetricName(metric.name),
          values: [metric.evaluation_a.median, metric.evaluation_b.median],
        }));
        break;
    }
  }

  private generateTableDataFromMetrics(): void {
    this.tableColumns = ['Evaluation A', 'Evaluation B', 'Difference'];
    this.tableData = this.metricDifferences.map((metric) => ({
      metric: this.formatMetricName(metric.metric_name),
      values: [
        metric.evaluation_a_value || 0,
        metric.evaluation_b_value || 0,
        metric.absolute_difference || 0,
      ],
    }));
  }

  // Public methods
  retryChart(): void {
    this.error = null;
    this.showFallback = false;
    this.initializeChart();
  }

  showTableFallback(): void {
    this.showFallback = true;
    this.error = null;
  }

  getChartAriaLabel(): string {
    const chartType = this.visualizationType;
    let metricCount = 0;

    if (this.visualizationData) {
      switch (this.visualizationData.type) {
        case 'radar':
          metricCount = (this.visualizationData as RadarVisualizationData)
            .labels.length;
          break;
        case 'bar':
          metricCount = (this.visualizationData as BarVisualizationData)
            .categories.length;
          break;
        case 'line':
          metricCount = (this.visualizationData as LineVisualizationData)
            .metrics.length;
          break;
      }
    } else {
      metricCount = this.metricDifferences.length;
    }

    return `${chartType} chart comparing ${metricCount} metrics between evaluations`;
  }

  // Helper methods
  trackByMetricName(
    index: number,
    row: { metric: string; values: number[] }
  ): string {
    return row.metric || index.toString();
  }

  formatMetricName(name: string | undefined): string {
    if (!name) return 'Unknown';
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase())
      .replace(/Deepeval/g, 'DeepEval')
      .replace(/G Eval/g, 'G-Eval');
  }

  formatNumber(value: number | undefined, decimals: number = 3): string {
    if (value === undefined || value === null || isNaN(value)) return 'N/A';
    return value.toFixed(decimals);
  }
}
