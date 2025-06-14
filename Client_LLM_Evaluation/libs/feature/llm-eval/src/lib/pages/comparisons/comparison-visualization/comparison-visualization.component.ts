import {
  Component,
  Input,
  OnChanges,
  SimpleChanges,
  ChangeDetectorRef,
  ElementRef,
  ViewChild,
  AfterViewInit,
  OnDestroy,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  MetricDifference,
  VisualizationData,
} from '@ngtx-apps/data-access/models';

// Declare Chart.js as external to avoid import issues
declare var Chart: any;

interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string | string[];
  borderColor?: string | string[];
  fill?: boolean;
  tension?: number;
  pointRadius?: number;
  pointHoverRadius?: number;
}

@Component({
  selector: 'app-comparison-visualization',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './comparison-visualization.component.html',
  styleUrls: ['./comparison-visualization.component.scss'],
})
export class ComparisonVisualizationComponent
  implements OnChanges, AfterViewInit, OnDestroy
{
  @Input() visualizationData: VisualizationData | null = null;
  @Input() visualizationType: 'radar' | 'bar' | 'line' = 'radar';
  @Input() metricDifferences: MetricDifference[] = [];

  @ViewChild('chartCanvas', { static: false })
  chartCanvas!: ElementRef<HTMLCanvasElement>;

  private chart: any = null;
  private isViewInitialized = false;
  private chartJsLoaded = false;

  // QA Theme Colors
  private readonly qaColors = {
    primary: '#8d0c4a',
    primaryLight: '#8e2157',
    success: '#28a745',
    error: '#dc3545',
    warning: '#ffc107',
    info: '#17a2b8',
    neutral: '#6c757d',
  };

  constructor(private cdr: ChangeDetectorRef) {}

  ngAfterViewInit(): void {
    this.isViewInitialized = true;
    this.loadChartJs().then(() => {
      this.createChart();
    });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (
      (changes['visualizationData'] ||
        changes['metricDifferences'] ||
        changes['visualizationType']) &&
      this.isViewInitialized &&
      this.chartJsLoaded
    ) {
      this.createChart();
    }
  }

  ngOnDestroy(): void {
    this.destroyChart();
  }

  /**
   * Load Chart.js from CDN if not available
   */
  private async loadChartJs(): Promise<void> {
    if (typeof Chart !== 'undefined') {
      this.chartJsLoaded = true;
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src =
        'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js';
      script.onload = () => {
        this.chartJsLoaded = true;
        resolve();
      };
      script.onerror = () => {
        console.error('Failed to load Chart.js');
        reject(new Error('Failed to load Chart.js'));
      };
      document.head.appendChild(script);
    });
  }

  /**
   * Create or update the chart
   */
  private createChart(): void {
    if (!this.chartCanvas?.nativeElement || !this.chartJsLoaded) return;

    this.destroyChart();

    const chartData = this.prepareChartData();
    if (!chartData) return;

    const config = this.getChartConfiguration(chartData);

    try {
      this.chart = new Chart(this.chartCanvas.nativeElement, config);
    } catch (error) {
      console.error('Error creating chart:', error);
      // Fallback to simple visualization
      this.createFallbackVisualization();
    }
  }

  /**
   * Create fallback visualization if Chart.js fails
   */
  private createFallbackVisualization(): void {
    if (!this.chartCanvas?.nativeElement) return;

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(
      0,
      0,
      this.chartCanvas.nativeElement.width,
      this.chartCanvas.nativeElement.height
    );

    // Draw simple bar chart as fallback
    ctx.fillStyle = this.qaColors.primary;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(
      'Chart visualization loading...',
      this.chartCanvas.nativeElement.width / 2,
      this.chartCanvas.nativeElement.height / 2
    );
  }

  /**
   * Destroy existing chart
   */
  private destroyChart(): void {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }

  /**
   * Prepare chart data from input
   */
  private prepareChartData(): {
    labels: string[];
    datasets: ChartDataset[];
  } | null {
    let labels: string[] = [];
    let datasets: ChartDataset[] = [];

    if (this.visualizationData) {
      labels = this.visualizationData.labels;
      datasets = this.processDatasets(this.visualizationData.datasets);
    } else if (this.metricDifferences.length > 0) {
      labels = this.metricDifferences.map((m) =>
        this.formatMetricName(m.metric_name)
      );
      datasets = this.createDatasetsFromMetrics();
    }

    return labels.length > 0 && datasets.length > 0
      ? { labels, datasets }
      : null;
  }

  /**
   * Create datasets from metric differences
   */
  private createDatasetsFromMetrics(): ChartDataset[] {
    if (
      this.visualizationType === 'radar' ||
      this.visualizationType === 'line'
    ) {
      return [
        {
          label: 'Evaluation A',
          data: this.metricDifferences.map((m) => m.evaluation_a_value),
          backgroundColor:
            this.visualizationType === 'radar'
              ? this.hexToRgba(this.qaColors.primary, 0.2)
              : this.qaColors.primary,
          borderColor: this.qaColors.primary,
          fill: this.visualizationType === 'radar',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
        {
          label: 'Evaluation B',
          data: this.metricDifferences.map((m) => m.evaluation_b_value),
          backgroundColor:
            this.visualizationType === 'radar'
              ? this.hexToRgba(this.qaColors.primaryLight, 0.2)
              : this.qaColors.primaryLight,
          borderColor: this.qaColors.primaryLight,
          fill: this.visualizationType === 'radar',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
      ];
    } else {
      // Bar chart
      const improvementColors = this.metricDifferences.map((m) =>
        m.is_improvement ? this.qaColors.success : this.qaColors.error
      );

      return [
        {
          label: 'Evaluation A',
          data: this.metricDifferences.map((m) => m.evaluation_a_value),
          backgroundColor: this.hexToRgba(this.qaColors.primary, 0.8),
          borderColor: this.qaColors.primary,
        },
        {
          label: 'Evaluation B',
          data: this.metricDifferences.map((m) => m.evaluation_b_value),
          backgroundColor: this.hexToRgba(this.qaColors.primaryLight, 0.8),
          borderColor: this.qaColors.primaryLight,
        },
        {
          label: 'Percentage Change',
          data: this.metricDifferences.map((m) => m.percentage_change || 0),
          backgroundColor: improvementColors,
          borderColor: improvementColors,
        },
      ];
    }
  }

  /**
   * Process datasets with QA theme colors
   */
  private processDatasets(datasets: any[]): ChartDataset[] {
    return datasets.map((dataset, index) => {
      const qaDataset: ChartDataset = {
        label: dataset.label,
        data: dataset.data,
        fill: dataset.fill || false,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 6,
      };

      // Apply QA theme colors
      if (index === 0) {
        qaDataset.backgroundColor =
          this.visualizationType === 'radar' ||
          this.visualizationType === 'line'
            ? this.hexToRgba(this.qaColors.primary, 0.2)
            : this.hexToRgba(this.qaColors.primary, 0.8);
        qaDataset.borderColor = this.qaColors.primary;
      } else if (index === 1) {
        qaDataset.backgroundColor =
          this.visualizationType === 'radar' ||
          this.visualizationType === 'line'
            ? this.hexToRgba(this.qaColors.primaryLight, 0.2)
            : this.hexToRgba(this.qaColors.primaryLight, 0.8);
        qaDataset.borderColor = this.qaColors.primaryLight;
      } else {
        // For difference datasets, use improvement/regression colors
        const colors = Array.isArray(dataset.data)
          ? dataset.data.map((value: number) =>
              value >= 0 ? this.qaColors.success : this.qaColors.error
            )
          : [this.qaColors.info];
        qaDataset.backgroundColor = colors;
        qaDataset.borderColor = colors;
      }

      return qaDataset;
    });
  }

  /**
   * Get chart configuration based on type
   */
  private getChartConfiguration(data: {
    labels: string[];
    datasets: ChartDataset[];
  }): any {
    const baseConfig: any = {
      type: this.getChartType(),
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 20,
              font: {
                size: 12,
                family: 'Arial, sans-serif',
              },
              color: '#212121', // qa-black
            },
          },
          tooltip: {
            backgroundColor: 'rgba(33, 33, 33, 0.9)', // qa-black with transparency
            titleColor: '#ffffff',
            bodyColor: '#ffffff',
            borderColor: this.qaColors.primary,
            borderWidth: 1,
            cornerRadius: 8,
            callbacks: {
              label: (context: any) => {
                const label = context.dataset.label || '';
                const value =
                  typeof context.parsed === 'object'
                    ? context.parsed.y || context.parsed.r || 0
                    : context.parsed;

                if (label === 'Percentage Change') {
                  return `${label}: ${value.toFixed(1)}%`;
                }
                return `${label}: ${value.toFixed(3)}`;
              },
            },
          },
        },
      },
    };

    // Add type-specific configurations
    switch (this.visualizationType) {
      case 'radar':
        this.configureRadarChart(baseConfig);
        break;
      case 'bar':
        this.configureBarChart(baseConfig);
        break;
      case 'line':
        this.configureLineChart(baseConfig);
        break;
    }

    return baseConfig;
  }

  /**
   * Configure radar chart specific options
   */
  private configureRadarChart(config: any): void {
    if (config.options) {
      config.options.scales = {
        r: {
          beginAtZero: true,
          max: 1,
          grid: {
            color: 'rgba(141, 12, 74, 0.1)', // qa-primary with transparency
          },
          angleLines: {
            color: 'rgba(141, 12, 74, 0.2)',
          },
          pointLabels: {
            font: {
              size: 11,
            },
            color: '#212121', // qa-black
          },
          ticks: {
            font: {
              size: 10,
            },
            color: '#757575', // text-secondary
            stepSize: 0.2,
          },
        },
      };
    }
  }

  /**
   * Configure bar chart specific options
   */
  private configureBarChart(config: any): void {
    if (config.options) {
      config.options.scales = {
        x: {
          grid: {
            display: false,
          },
          ticks: {
            font: {
              size: 10,
            },
            color: '#212121', // qa-black
            maxRotation: 45,
          },
        },
        y: {
          beginAtZero: true,
          grid: {
            color: 'rgba(141, 12, 74, 0.1)',
          },
          ticks: {
            font: {
              size: 10,
            },
            color: '#757575', // text-secondary
          },
        },
      };

      // Add interaction for grouped bars
      config.options.interaction = {
        mode: 'index',
        intersect: false,
      };
    }
  }

  /**
   * Configure line chart specific options
   */
  private configureLineChart(config: any): void {
    if (config.options) {
      config.options.scales = {
        x: {
          grid: {
            color: 'rgba(141, 12, 74, 0.1)',
          },
          ticks: {
            font: {
              size: 10,
            },
            color: '#212121', // qa-black
            maxRotation: 45,
          },
        },
        y: {
          beginAtZero: true,
          grid: {
            color: 'rgba(141, 12, 74, 0.1)',
          },
          ticks: {
            font: {
              size: 10,
            },
            color: '#757575', // text-secondary
          },
        },
      };

      config.options.elements = {
        line: {
          tension: 0.4,
        },
        point: {
          radius: 4,
          hoverRadius: 6,
        },
      };
    }
  }

  /**
   * Get Chart.js chart type
   */
  private getChartType(): string {
    switch (this.visualizationType) {
      case 'radar':
        return 'radar';
      case 'bar':
        return 'bar';
      case 'line':
        return 'line';
      default:
        return 'radar';
    }
  }

  /**
   * Format metric name for display
   */
  private formatMetricName(name: string | undefined): string {
    if (!name) return 'Unknown Metric';

    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase())
      .replace(/Deepeval/g, 'DeepEval')
      .replace(/G Eval/g, 'G-Eval');
  }

  /**
   * Convert hex color to rgba
   */
  private hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  /**
   * Check if data is available for visualization
   */
  hasData(): boolean {
    return (
      (this.visualizationData?.datasets?.length || 0) > 0 ||
      this.metricDifferences.length > 0
    );
  }

  /**
   * Get chart title based on type
   */
  getChartTitle(): string {
    switch (this.visualizationType) {
      case 'radar':
        return 'Metric Comparison Radar';
      case 'bar':
        return 'Metric Comparison Bar Chart';
      case 'line':
        return 'Metric Trends';
      default:
        return 'Metric Comparison';
    }
  }

  /**
   * Get chart statistics
   */
  getChartStats(): {
    metricsCount: number;
    evaluationsCount: number;
    maxValue: number;
    minValue: number;
  } {
    if (!this.hasData()) {
      return { metricsCount: 0, evaluationsCount: 0, maxValue: 0, minValue: 0 };
    }

    const datasets = this.visualizationData?.datasets || [];
    const metricsCount =
      this.metricDifferences.length ||
      this.visualizationData?.labels?.length ||
      0;
    const evaluationsCount = Math.min(datasets.length, 2); // Only count first two datasets as evaluations

    // Calculate min/max from metric differences if available
    if (this.metricDifferences.length > 0) {
      const allValues = [
        ...this.metricDifferences.map((m) => m.evaluation_a_value),
        ...this.metricDifferences.map((m) => m.evaluation_b_value),
      ].filter((v) => !isNaN(v) && isFinite(v));

      return {
        metricsCount,
        evaluationsCount,
        maxValue: allValues.length > 0 ? Math.max(...allValues) : 0,
        minValue: allValues.length > 0 ? Math.min(...allValues) : 0,
      };
    }

    // Fallback to visualization data
    const allValues = datasets
      .flatMap((d) => d.data || [])
      .filter((v) => !isNaN(v) && isFinite(v));

    return {
      metricsCount,
      evaluationsCount,
      maxValue: allValues.length > 0 ? Math.max(...allValues) : 0,
      minValue: allValues.length > 0 ? Math.min(...allValues) : 0,
    };
  }
}
