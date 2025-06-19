import {
  Component,
  OnInit,
  OnDestroy,
  ChangeDetectorRef,
  NO_ERRORS_SCHEMA,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, takeUntil, of } from 'rxjs';
import { finalize, catchError } from 'rxjs/operators';

import {
  Comparison,
  ComparisonDetail,
  ComparisonStatus,
  MetricDifference,
  SampleDifference,
  VisualizationData,
  ApiVisualizationData,
} from '@ngtx-apps/data-access/models';
import { ComparisonService } from '@ngtx-apps/data-access/services';
import {
  ConfirmationDialogService,
  NotificationService,
} from '@ngtx-apps/utils/services';
import { ComparisonVisualizationComponent } from '../comparison-visualization/comparison-visualization.component';

interface EvaluationInfo {
  id: string;
  name: string;
  agentName: string;
  agentId: string;
  datasetName: string;
  datasetId: string;
  promptName: string;
  promptId: string;
  method: string;
  processedItems: number;
  overallScore: number;
  status: string;
  duration?: string;
}

interface ComparisonSummary {
  overallPerformanceChange: number;
  overallStatus: 'improved' | 'regressed' | 'unchanged';
  totalMetrics: number;
  improvedMetrics: number;
  regressedMetrics: number;
  unchangedMetrics: number;
  samplesAnalyzed: number;
  statisticalPower: string;
  hasSignificantChanges: boolean;
}

interface ParsedInsight {
  type: 'header' | 'paragraph' | 'list' | 'conclusion';
  title?: string;
  content?: string;
  items?: string[];
}

interface FormattedNarrativeInsights {
  sections: ParsedInsight[];
  hasContent: boolean;
}

@Component({
  selector: 'app-comparison-detail',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    RouterModule,
    ComparisonVisualizationComponent,
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './comparison-detail.page.html',
  styleUrls: ['./comparison-detail.page.scss'],
})
export class ComparisonDetailPage implements OnInit, OnDestroy {
  comparison: ComparisonDetail | null = null;
  comparisonId: string = '';
  isLoading = false;
  error: string | null = null;

  // Enhanced evaluation information
  evaluationA: EvaluationInfo | null = null;
  evaluationB: EvaluationInfo | null = null;

  // Processed summary data
  summaryData: ComparisonSummary | null = null;

  // Tab state - simplified
  selectedTabIndex = 0;
  readonly tabs = [
    { id: 'metrics', label: 'Metrics', icon: 'trending-up' },
    { id: 'charts', label: 'Charts', icon: 'bar-chart' },
    { id: 'samples', label: 'Samples', icon: 'list' },
    { id: 'details', label: 'Details', icon: 'info' },
  ];

  // Metrics data
  metricDifferences: MetricDifference[] = [];
  topImprovements: Array<{ metric: string; value: number; impact: string }> =
    [];
  topRegressions: Array<{ metric: string; value: number; impact: string }> = [];

  // Visualization data - fix the type to match what the visualization component expects
  selectedVisualization: 'radar' | 'bar' | 'line' = 'radar';
  visualizationData: ApiVisualizationData | null = null;
  isLoadingVisualization: boolean = false;

  // Sample data
  sampleDifferences: SampleDifference[] = [];
  filteredSamples: SampleDifference[] = [];
  sampleFilter: string = 'all';
  sampleSort: string = 'difference';

  // UI state
  showInsights = false;
  showSampleDetails = false;
  selectedSample: SampleDifference | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private comparisonService: ComparisonService,
    private confirmationDialogService: ConfirmationDialogService,
    private notificationService: NotificationService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.route.paramMap.pipe(takeUntil(this.destroy$)).subscribe((params) => {
      const id = params.get('id');
      if (id) {
        this.comparisonId = id;
        this.loadComparison(id);
      } else {
        this.error = 'Comparison ID not found.';
        this.cdr.markForCheck();
      }
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Load comparison details
   */
  loadComparison(id: string): void {
    this.isLoading = true;
    this.error = null;
    this.cdr.markForCheck();

    this.comparisonService
      .getComparison(id)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isLoading = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe({
        next: (comparison) => {
          this.comparison = comparison;
          this.processComparisonData(comparison);
          this.loadEnhancedMetadata(comparison);
          this.cdr.markForCheck();
        },
        error: (err) => {
          this.error = 'Failed to load comparison details. Please try again.';
          console.error('Error loading comparison:', err);
          this.cdr.markForCheck();
        },
      });
  }

  /**
   * Load enhanced metadata for evaluations
   */
  private loadEnhancedMetadata(comparison: ComparisonDetail): void {
    // Create evaluation info from available comparison data
    if (comparison.evaluation_a) {
      this.evaluationA = this.createEvaluationInfoFromData(
        comparison.evaluation_a,
        'a'
      );
    } else {
      this.evaluationA = this.createFallbackEvaluationInfo(
        comparison.evaluation_a_id,
        'a'
      );
    }

    if (comparison.evaluation_b) {
      this.evaluationB = this.createEvaluationInfoFromData(
        comparison.evaluation_b,
        'b'
      );
    } else {
      this.evaluationB = this.createFallbackEvaluationInfo(
        comparison.evaluation_b_id,
        'b'
      );
    }

    this.cdr.markForCheck();
  }

  /**
   * Create evaluation info from available data
   */
  private createEvaluationInfoFromData(
    evaluationData: any,
    type: 'a' | 'b'
  ): EvaluationInfo {
    return {
      id: evaluationData.id || '',
      name: evaluationData.name || `Evaluation ${type.toUpperCase()}`,
      agentName:
        evaluationData.agent_name ||
        this.truncateId(evaluationData.agent_id || 'Unknown'),
      agentId: evaluationData.agent_id || '',
      datasetName:
        evaluationData.dataset_name ||
        this.truncateId(evaluationData.dataset_id || 'Unknown'),
      datasetId: evaluationData.dataset_id || '',
      promptName:
        evaluationData.prompt_name ||
        this.truncateId(evaluationData.prompt_id || 'Default'),
      promptId: evaluationData.prompt_id || '',
      method: evaluationData.method || 'Unknown',
      processedItems: evaluationData.processed_items || 0,
      overallScore: this.getOverallScore(type),
      status: evaluationData.status || 'unknown',
      duration: this.calculateDuration(
        evaluationData.start_time,
        evaluationData.end_time
      ),
    };
  }

  /**
   * Create fallback evaluation info when detailed data isn't available
   */
  private createFallbackEvaluationInfo(
    id: string,
    type: 'a' | 'b'
  ): EvaluationInfo {
    return {
      id: id,
      name: `Evaluation ${type.toUpperCase()}`,
      agentName: this.truncateId(id),
      agentId: id,
      datasetName: 'Loading...',
      datasetId: '',
      promptName: 'Loading...',
      promptId: '',
      method: 'Unknown',
      processedItems: 0,
      overallScore: this.getOverallScore(type),
      status: 'unknown',
      duration: 'Unknown',
    };
  }

  /**
   * Truncate long IDs for display
   */
  private truncateId(id: string): string {
    if (!id || id.length <= 8) return id;
    return id.substring(0, 8) + '...';
  }

  /**
   * Calculate duration between start and end times
   */
  private calculateDuration(startTime?: string, endTime?: string): string {
    if (!startTime || !endTime) return 'Unknown';

    try {
      const start = new Date(startTime);
      const end = new Date(endTime);
      const diffMs = end.getTime() - start.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      const diffSecs = Math.floor((diffMs % 60000) / 1000);

      if (diffMins > 0) {
        return `${diffMins}m ${diffSecs}s`;
      }
      return `${diffSecs}s`;
    } catch (error) {
      return 'Unknown';
    }
  }

  /**
   * Process comparison data and extract meaningful insights
   */
  processComparisonData(comparison: ComparisonDetail): void {
    // Process metrics
    this.metricDifferences = comparison.metric_differences || [];

    // Create summary data
    this.summaryData = this.createSummaryData(comparison);

    // Process top changes
    this.processTopChanges();

    // Process sample data
    this.processSampleData(comparison);

    // Load initial visualization
    this.loadVisualizationData(this.selectedVisualization);
  }

  /**
   * Create enhanced summary data
   */
  private createSummaryData(comparison: ComparisonDetail): ComparisonSummary {
    const summary = comparison.summary;
    const overallScores =
      comparison.comparison_results?.['overall_comparison']?.overall_scores;

    const overallChange = overallScores?.percentage_change || 0;
    const totalMetrics = summary?.total_metrics || 0;
    const improved = summary?.improved_metrics || 0;
    const regressed = summary?.regressed_metrics || 0;
    const unchanged = this.getMaxValue(0, totalMetrics - improved - regressed);

    return {
      overallPerformanceChange: overallChange,
      overallStatus:
        overallChange > 0
          ? 'improved'
          : overallChange < 0
          ? 'regressed'
          : 'unchanged',
      totalMetrics,
      improvedMetrics: improved,
      regressedMetrics: regressed,
      unchangedMetrics: unchanged,
      samplesAnalyzed: summary?.matched_samples || 0,
      statisticalPower: summary?.statistical_power?.power_category || 'Unknown',
      hasSignificantChanges:
        (summary?.significant_improvements || 0) +
          (summary?.significant_regressions || 0) >
        0,
    };
  }

  /**
   * Process top changes for simplified display
   */
  private processTopChanges(): void {
    this.topImprovements = this.metricDifferences
      .filter((m) => m.is_improvement && (m.percentage_change || 0) > 0)
      .sort((a, b) => (b.percentage_change || 0) - (a.percentage_change || 0))
      .slice(0, 3)
      .map((m) => ({
        metric: this.getFormattedMetricName(m.metric_name),
        value: m.percentage_change || 0,
        impact: this.getImpactLevel(
          this.getAbsoluteValue(m.percentage_change || 0)
        ),
      }));

    this.topRegressions = this.metricDifferences
      .filter((m) => !m.is_improvement && (m.percentage_change || 0) < 0)
      .sort((a, b) => (a.percentage_change || 0) - (b.percentage_change || 0))
      .slice(0, 3)
      .map((m) => ({
        metric: this.getFormattedMetricName(m.metric_name),
        value: m.percentage_change || 0,
        impact: this.getImpactLevel(
          this.getAbsoluteValue(m.percentage_change || 0)
        ),
      }));
  }

  /**
   * Get impact level from percentage change
   */
  private getImpactLevel(percentage: number): string {
    const absPercentage = this.getAbsoluteValue(percentage);
    if (absPercentage >= 20) return 'High';
    if (absPercentage >= 5) return 'Medium';
    return 'Low';
  }

  /**
   * Process sample data from API response
   */
  processSampleData(comparison: ComparisonDetail): void {
    this.sampleDifferences = [];

    if (comparison.result_differences) {
      this.sampleDifferences = Object.entries(comparison.result_differences)
        .map(([sampleId, sampleData]) => {
          if ((sampleData as any).comparison?.missing_in) {
            return null;
          }

          const data = sampleData as any;
          return {
            sample_id: sampleId,
            evaluation_a_score: data.evaluation_a?.overall_score,
            evaluation_b_score: data.evaluation_b?.overall_score,
            absolute_difference: data.comparison?.absolute_difference,
            percentage_difference: data.comparison?.percentage_change,
            status: this.determineSampleStatus(data.comparison),
            input_data: {},
            evaluation_a_output: data.evaluation_a || {},
            evaluation_b_output: data.evaluation_b || {},
          };
        })
        .filter((sample) => sample !== null) as SampleDifference[];
    }

    this.filterSamples();
  }

  /**
   * Determine sample status from comparison data
   */
  private determineSampleStatus(comparison: any): string {
    if (!comparison) return 'unchanged';
    if (comparison.is_improvement === true) return 'improved';
    if (
      comparison.is_improvement === false &&
      comparison.absolute_difference !== 0
    )
      return 'regressed';
    return 'unchanged';
  }

  /**
   * Load visualization data with better error handling and real API support
   */
  loadVisualizationData(type: 'radar' | 'bar' | 'line'): void {
    if (!this.comparisonId || !this.hasResults()) {
      return;
    }

    this.isLoadingVisualization = true;
    this.visualizationData = null;

    // Use the enhanced service method that handles real API formats
    this.comparisonService
      .getVisualizationDataWithFallback(this.comparisonId, type)
      .pipe(
        takeUntil(this.destroy$),
        catchError((error) => {
          console.error(`Error loading ${type} visualization:`, error);
          // Return null to trigger fallback behavior in component
          return of(null);
        }),
        finalize(() => {
          this.isLoadingVisualization = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe((data: ApiVisualizationData | VisualizationData | null) => {
        if (data) {
          // Convert VisualizationData to ApiVisualizationData format if needed
          this.visualizationData = this.convertToApiFormat(data, type);
        } else {
          console.warn(`No ${type} visualization data available`);
          this.visualizationData = null;
        }
        this.cdr.markForCheck();
      });
  }

  /**
   * Convert VisualizationData to ApiVisualizationData format
   */
  private convertToApiFormat(
    data: ApiVisualizationData | VisualizationData,
    type: 'radar' | 'bar' | 'line'
  ): ApiVisualizationData | null {
    // If it's already ApiVisualizationData, return as is
    if (this.isApiVisualizationData(data)) {
      return data;
    }

    // Convert VisualizationData to ApiVisualizationData
    const vizData = data as VisualizationData;

    switch (type) {
      case 'radar':
        return {
          type: 'radar',
          labels: vizData.labels,
          series: vizData.datasets.map((dataset) => ({
            name: dataset.label,
            data: dataset.data as number[],
          })),
          is_inverted: vizData.labels.map(() => false),
        };

      case 'bar':
        return {
          type: 'bar',
          categories: vizData.labels,
          series: vizData.datasets.map((dataset) => ({
            name: dataset.label,
            data: dataset.data as number[],
            type: dataset.type === 'line' ? 'line' : 'bar',
          })),
          is_significant: vizData.labels.map(() => false),
          higher_is_better: vizData.labels.map(() => true),
        };

      case 'line':
        // For line charts, we need to create a different structure
        return {
          type: 'line',
          metrics: vizData.labels.map((label, index) => ({
            name: label,
            higher_is_better: true,
            is_significant: false,
            evaluation_a: {
              name: 'Evaluation A',
              values: [vizData.datasets[0]?.data[index] || 0] as number[],
              min: vizData.datasets[0]?.data[index] || 0,
              max: vizData.datasets[0]?.data[index] || 0,
              q1: vizData.datasets[0]?.data[index] || 0,
              q3: vizData.datasets[0]?.data[index] || 0,
              median: vizData.datasets[0]?.data[index] || 0,
            },
            evaluation_b: {
              name: 'Evaluation B',
              values: [vizData.datasets[1]?.data[index] || 0] as number[],
              min: vizData.datasets[1]?.data[index] || 0,
              max: vizData.datasets[1]?.data[index] || 0,
              q1: vizData.datasets[1]?.data[index] || 0,
              q3: vizData.datasets[1]?.data[index] || 0,
              median: vizData.datasets[1]?.data[index] || 0,
            },
          })),
        };

      default:
        return null;
    }
  }

  /**
   * Type guard to check if data is ApiVisualizationData
   */
  private isApiVisualizationData(data: any): data is ApiVisualizationData {
    if (!data || !data.type) return false;

    switch (data.type) {
      case 'radar':
        return !!(data.labels && data.series && data.is_inverted);
      case 'bar':
        return !!(data.categories && data.series && data.is_significant);
      case 'line':
        return !!data.metrics;
      default:
        return false;
    }
  }

  /**
   * Filter and sort samples
   */
  filterSamples(): void {
    if (!this.sampleDifferences || this.sampleDifferences.length === 0) {
      this.filteredSamples = [];
      return;
    }

    let filtered = [...this.sampleDifferences];

    if (this.sampleFilter !== 'all') {
      filtered = filtered.filter(
        (sample) => sample.status === this.sampleFilter
      );
    }

    switch (this.sampleSort) {
      case 'difference':
        filtered.sort((a, b) => {
          const diffA = this.getAbsoluteValue(a.absolute_difference || 0);
          const diffB = this.getAbsoluteValue(b.absolute_difference || 0);
          return diffB - diffA;
        });
        break;
      case 'id':
        filtered.sort((a, b) => a.sample_id.localeCompare(b.sample_id));
        break;
      case 'score_a':
        filtered.sort(
          (a, b) => (b.evaluation_a_score || 0) - (a.evaluation_a_score || 0)
        );
        break;
      case 'score_b':
        filtered.sort(
          (a, b) => (b.evaluation_b_score || 0) - (a.evaluation_b_score || 0)
        );
        break;
    }

    this.filteredSamples = filtered;
    this.cdr.markForCheck();
  }

  /**
   * Handle tab selection
   */
  selectTab(index: number): void {
    this.selectedTabIndex = index;

    // Load visualization data when charts tab is selected
    if (index === 1) {
      this.loadVisualizationData(this.selectedVisualization);
    }

    this.cdr.markForCheck();
  }

  /**
   * Handle visualization type selection
   */
  selectVisualization(type: 'radar' | 'bar' | 'line'): void {
    this.selectedVisualization = type;
    this.loadVisualizationData(type);
  }

  /**
   * Toggle insights display
   */
  toggleInsights(): void {
    this.showInsights = !this.showInsights;
    this.cdr.markForCheck();
  }

  /**
   * Get overall score for evaluation
   */
  getOverallScore(evaluation: 'a' | 'b'): number {
    if (
      !this.comparison?.comparison_results?.['overall_comparison']
        ?.overall_scores
    ) {
      return 0;
    }

    const scores =
      this.comparison.comparison_results['overall_comparison'].overall_scores;
    return evaluation === 'a' ? scores.evaluation_a : scores.evaluation_b;
  }

  /**
   * Get formatted metric name
   */
  getFormattedMetricName(name: string | undefined): string {
    if (!name) return 'Unknown Metric';

    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase())
      .replace(/Deepeval/g, 'DeepEval')
      .replace(/G Eval/g, 'G-Eval');
  }

  /**
   * Check if comparison has results
   */
  hasResults(): boolean {
    return (
      !!this.comparison?.summary &&
      this.comparison.status === ComparisonStatus.COMPLETED
    );
  }

  /**
   * Check if comparison can be run
   */
  canRunComparison(): boolean {
    return (
      this.comparison?.status === ComparisonStatus.PENDING ||
      this.comparison?.status === ComparisonStatus.FAILED
    );
  }

  /**
   * Get status badge class
   */
  getStatusClass(status: ComparisonStatus): string {
    switch (status) {
      case ComparisonStatus.COMPLETED:
        return 'completed';
      case ComparisonStatus.RUNNING:
        return 'running';
      case ComparisonStatus.PENDING:
        return 'pending';
      case ComparisonStatus.FAILED:
        return 'failed';
      default:
        return 'neutral';
    }
  }

  /**
   * Format date for display
   */
  formatDate(dateString: string | undefined): string {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      }).format(date);
    } catch (e) {
      return 'Invalid date';
    }
  }

  /**
   * Format percentage with proper sign
   */
  formatPercentage(value: number | undefined): string {
    if (value === undefined || value === null) return '0.0%';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  }

  /**
   * Format number with proper precision
   */
  formatNumber(value: number | undefined, decimals: number = 3): string {
    if (value === undefined || value === null) return 'N/A';
    return value.toFixed(decimals);
  }

  // Helper methods for template
  /**
   * Get absolute value - helper for template
   */
  getAbsoluteValue(value: number): number {
    return Math.abs(value);
  }

  /**
   * Get maximum value - helper for template
   */
  getMaxValue(a: number, b: number): number {
    return Math.max(a, b);
  }

  /**
   * Get impact level for template use
   */
  getImpactLevelForValue(value: number): string {
    return this.getImpactLevel(Math.abs(value));
  }

  getImpactClass(percentage: number): string {
    const absPercentage = this.getAbsoluteValue(percentage);
    if (absPercentage >= 20) return 'high';
    if (absPercentage >= 5) return 'medium';
    return 'low';
  }

  // Action methods
  runComparison(): void {
    if (!this.comparisonId) return;

    this.confirmationDialogService
      .confirm({
        title: 'Run Comparison',
        message: 'Are you sure you want to run this comparison?',
        confirmText: 'Run',
        cancelText: 'Cancel',
        type: 'info',
      })
      .subscribe((confirmed) => {
        if (confirmed) {
          this.isLoading = true;
          this.cdr.markForCheck();

          this.comparisonService
            .runComparison(this.comparisonId)
            .pipe(
              takeUntil(this.destroy$),
              finalize(() => {
                this.isLoading = false;
                this.cdr.markForCheck();
              })
            )
            .subscribe({
              next: () => {
                this.notificationService.success('Comparison is running');
                setTimeout(() => this.loadComparison(this.comparisonId), 2000);
              },
              error: (error) => {
                this.notificationService.error(
                  'Failed to run comparison. Please try again.'
                );
                console.error('Error running comparison:', error);
              },
            });
        }
      });
  }

  editComparison(): void {
    if (this.comparison) {
      this.router.navigate(['app/comparisons', this.comparison.id, 'edit']);
    }
  }

  deleteComparison(): void {
    if (!this.comparison) return;

    this.confirmationDialogService
      .confirmDelete('Comparison')
      .subscribe((confirmed) => {
        if (confirmed) {
          this.comparisonService
            .deleteComparison(this.comparison!.id)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.notificationService.success(
                  'Comparison deleted successfully'
                );
                this.router.navigate(['app/comparisons']);
              },
              error: (error) => {
                this.notificationService.error(
                  'Failed to delete comparison. Please try again.'
                );
                console.error('Error deleting comparison:', error);
              },
            });
        }
      });
  }

  viewEvaluation(evaluationId: string): void {
    if (evaluationId) {
      this.router.navigate(['app/evaluations', evaluationId]);
    }
  }

  viewSampleDetails(sample: SampleDifference): void {
    this.selectedSample = sample;
    this.showSampleDetails = true;
    this.cdr.markForCheck();
  }

  closeSampleDetails(): void {
    this.showSampleDetails = false;
    this.selectedSample = null;
    this.cdr.markForCheck();
  }

  sortSamples(): void {
    this.filterSamples();
  }

  // Helper method to check if object has meaningful data
  hasObjectData(obj: any): boolean {
    if (!obj) return false;
    if (typeof obj === 'string') return obj.trim().length > 0;
    if (typeof obj === 'object') {
      return Object.keys(obj).length > 0;
    }
    return true;
  }

  // Helper methods for template
  getSampleStatusClass(status: string): string {
    switch (status) {
      case 'improved':
        return 'improved';
      case 'regressed':
        return 'regressed';
      case 'unchanged':
        return 'unchanged';
      default:
        return 'unchanged';
    }
  }

  getSampleDifferenceClass(sample: SampleDifference): string {
    if (sample.status === 'improved') return 'positive';
    if (sample.status === 'regressed') return 'negative';
    return 'neutral';
  }

  getMetricRowClass(metric: MetricDifference): string {
    return metric.is_improvement ? 'improved' : 'regressed';
  }

  getDifferenceClass(metric: MetricDifference): string {
    if (metric.is_improvement) return 'positive';
    if (metric.absolute_difference === 0) return 'neutral';
    return 'negative';
  }

  formatJsonData(data: any): string {
    if (!data) return 'No data available';
    try {
      return JSON.stringify(data, null, 2);
    } catch (e) {
      return String(data);
    }
  }
  /**
   * Parse and format narrative insights text
   */
  parseNarrativeInsights(rawText: string): FormattedNarrativeInsights {
    if (!rawText || !rawText.trim()) {
      return { sections: [], hasContent: false };
    }

    const sections: ParsedInsight[] = [];
    const lines = rawText
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    let currentSection: ParsedInsight | null = null;
    let currentListItems: string[] = [];

    for (const line of lines) {
      // Handle headers (bold text followed by colon)
      if (line.match(/^\*\*(.+?):\*\*(.*)$/)) {
        // Finish any current list
        if (currentSection && currentListItems.length > 0) {
          currentSection.items = [...currentListItems];
          currentListItems = [];
        }

        const match = line.match(/^\*\*(.+?):\*\*(.*)$/);
        if (match) {
          const title = match[1].trim();
          const content = match[2].trim();

          currentSection = {
            type: title.toLowerCase().includes('conclusion')
              ? 'conclusion'
              : 'header',
            title,
            content: this.cleanFormattedText(content),
          };
          sections.push(currentSection);
        }
      }
      // Handle list items
      else if (line.startsWith('- ')) {
        const listItem = line.substring(2).trim();
        currentListItems.push(this.cleanFormattedText(listItem));
      }
      // Handle regular paragraphs
      else if (line.length > 0) {
        // If we have pending list items, create a list section
        if (currentListItems.length > 0) {
          if (currentSection) {
            currentSection.items = [...currentListItems];
            currentListItems = [];
          }
        }

        // If this isn't part of a header, create a paragraph
        if (!line.startsWith('**')) {
          sections.push({
            type: 'paragraph',
            content: this.cleanFormattedText(line),
          });
        }
      }
    }

    // Handle any remaining list items
    if (currentSection && currentListItems.length > 0) {
      currentSection.items = [...currentListItems];
    }

    return { sections, hasContent: sections.length > 0 };
  }

  /**
   * Clean formatted text (remove markdown syntax, preserve content)
   */
  private cleanFormattedText(text: string): string {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold text
      .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic text
      .replace(/`(.*?)`/g, '<code>$1</code>') // Code text
      .trim();
  }

  /**
   * Get formatted narrative insights for template
   */
  getFormattedNarrativeInsights(): FormattedNarrativeInsights {
    if (!this.comparison?.narrative_insights) {
      return { sections: [], hasContent: false };
    }

    return this.parseNarrativeInsights(this.comparison.narrative_insights);
  }

  /**
   * Get insight section icon based on type
   */
  getInsightSectionIcon(section: ParsedInsight): string {
    switch (section.type) {
      case 'conclusion':
        return 'check-circle';
      case 'header':
        if (section.title?.toLowerCase().includes('improvement'))
          return 'trending-up';
        if (section.title?.toLowerCase().includes('regression'))
          return 'trending-down';
        if (section.title?.toLowerCase().includes('statistical'))
          return 'bar-chart-2';
        if (section.title?.toLowerCase().includes('sample')) return 'users';
        return 'info';
      default:
        return 'minus';
    }
  }

  /**
   * Get insight section class for styling
   */
  getInsightSectionClass(section: ParsedInsight): string {
    switch (section.type) {
      case 'conclusion':
        return 'insight-conclusion';
      case 'header':
        if (section.title?.toLowerCase().includes('improvement'))
          return 'insight-positive';
        if (section.title?.toLowerCase().includes('regression'))
          return 'insight-negative';
        return 'insight-info';
      default:
        return 'insight-neutral';
    }
  }
}
