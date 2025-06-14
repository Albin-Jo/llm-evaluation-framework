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
} from '@ngtx-apps/data-access/models';
import {
  ComparisonService,
  EvaluationService,
} from '@ngtx-apps/data-access/services';
import {
  ConfirmationDialogService,
  NotificationService,
} from '@ngtx-apps/utils/services';
import { ComparisonVisualizationComponent } from '../comparison-visualization/comparison-visualization.component';

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

  // Tab state
  selectedTabIndex = 0;

  // Metrics data - use API provided data directly
  metricDifferences: MetricDifference[] = [];

  // Visualization data
  selectedVisualization: 'radar' | 'bar' | 'line' = 'radar';
  visualizationData: VisualizationData | null = null;

  // Sample data - use API provided data directly
  sampleDifferences: SampleDifference[] = [];
  filteredSamples: SampleDifference[] = [];
  sampleFilter: string = 'all';
  sampleSort: string = 'difference';

  // Sample details modal
  showSampleDetails = false;
  selectedSample: SampleDifference | null = null;

  // Enhanced analysis summary state
  isNarrativeExpanded = false;

  // Math object for template access
  Math = Math;

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private comparisonService: ComparisonService,
    private evaluationService: EvaluationService,
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
          console.log('Loaded comparison:', comparison);
          this.comparison = comparison;
          this.processComparisonData(comparison);
          this.loadVisualizationData(this.selectedVisualization);
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
   * Process comparison data - use API provided data directly
   */
  processComparisonData(comparison: ComparisonDetail): void {
    // Use pre-processed metric differences from API
    this.metricDifferences = comparison.metric_differences || [];
    console.log('Using API metric differences:', this.metricDifferences);

    // Process sample differences from result_differences
    this.processSampleData(comparison);
  }

  /**
   * Process sample data from API response
   */
  processSampleData(comparison: ComparisonDetail): void {
    this.sampleDifferences = [];

    if (comparison.result_differences) {
      this.sampleDifferences = Object.entries(comparison.result_differences)
        .map(([sampleId, sampleData]) => {
          // Skip samples missing in one evaluation
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
    console.log('Processed sample differences:', this.sampleDifferences);
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
   * Load visualization data
   */
  loadVisualizationData(type: 'radar' | 'bar' | 'line'): void {
    if (!this.comparisonId || !this.hasResults()) return;

    this.comparisonService
      .getVisualizationData(this.comparisonId, type)
      .pipe(
        takeUntil(this.destroy$),
        catchError((error) => {
          console.error(`Error loading ${type} visualization:`, error);
          return of(this.createVisualizationDataFromMetrics(type));
        })
      )
      .subscribe((data) => {
        if (data) {
          this.visualizationData = data;
          this.cdr.markForCheck();
        }
      });
  }

  /**
   * Create visualization data from metric differences
   */
  private createVisualizationDataFromMetrics(
    type: 'radar' | 'bar' | 'line'
  ): VisualizationData {
    const labels = this.metricDifferences.map((m) =>
      this.getFormattedMetricName(m.metric_name)
    );

    let datasets;

    if (type === 'radar' || type === 'line') {
      datasets = [
        {
          label: this.getEvaluationName('a'),
          data: this.metricDifferences.map((m) => m.evaluation_a_value),
          backgroundColor: 'rgba(141, 12, 74, 0.2)', // qa-primary with transparency
          borderColor: 'rgba(141, 12, 74, 1)', // qa-primary
          fill: type === 'radar',
        },
        {
          label: this.getEvaluationName('b'),
          data: this.metricDifferences.map((m) => m.evaluation_b_value),
          backgroundColor: 'rgba(142, 33, 87, 0.2)', // qa-primary-light with transparency
          borderColor: 'rgba(142, 33, 87, 1)', // qa-primary-light
          fill: type === 'radar',
        },
      ];
    } else {
      datasets = [
        {
          label: this.getEvaluationName('a'),
          data: this.metricDifferences.map((m) => m.evaluation_a_value),
          backgroundColor: 'rgba(141, 12, 74, 0.8)', // qa-primary
        },
        {
          label: this.getEvaluationName('b'),
          data: this.metricDifferences.map((m) => m.evaluation_b_value),
          backgroundColor: 'rgba(142, 33, 87, 0.8)', // qa-primary-light
        },
        {
          label: 'Difference (%)',
          data: this.metricDifferences.map((m) => m.percentage_change || 0),
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
        },
      ];
    }

    return {
      type,
      labels,
      datasets,
    };
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
          const diffA = Math.abs(a.absolute_difference || 0);
          const diffB = Math.abs(b.absolute_difference || 0);
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
   * Sort samples
   */
  sortSamples(): void {
    this.filterSamples();
  }

  /**
   * Handle visualization type selection
   */
  selectVisualization(type: 'radar' | 'bar' | 'line'): void {
    this.selectedVisualization = type;
    this.loadVisualizationData(type);
  }

  /**
   * Show sample details
   */
  viewSampleDetails(sample: SampleDifference): void {
    this.selectedSample = sample;
    this.showSampleDetails = true;
    this.cdr.markForCheck();
  }

  /**
   * Close sample details modal
   */
  closeSampleDetails(): void {
    this.showSampleDetails = false;
    this.selectedSample = null;
    this.cdr.markForCheck();
  }

  /**
   * Run comparison
   */
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

  /**
   * Edit comparison
   */
  editComparison(): void {
    if (this.comparison) {
      this.router.navigate(['app/comparisons', this.comparison.id, 'edit']);
    }
  }

  /**
   * Delete comparison
   */
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

  /**
   * Navigate to evaluation
   */
  viewEvaluation(evaluationId: string): void {
    if (evaluationId) {
      this.router.navigate(['app/evaluations', evaluationId]);
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
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(date);
    } catch (e) {
      return 'Invalid date';
    }
  }

  /**
   * Get evaluation name - FIXED to use direct API data
   */
  getEvaluationName(evaluation: 'a' | 'b'): string {
    if (!this.comparison) return `Evaluation ${evaluation.toUpperCase()}`;

    return evaluation === 'a'
      ? this.comparison.evaluation_a?.['name'] ||
          this.comparison.summary?.evaluation_a_name ||
          'Evaluation A'
      : this.comparison.evaluation_b?.['name'] ||
          this.comparison.summary?.evaluation_b_name ||
          'Evaluation B';
  }

  /**
   * Get evaluation method - FIXED to use direct API data
   */
  getEvaluationMethod(evaluation: 'a' | 'b'): string {
    if (!this.comparison) return 'N/A';

    return evaluation === 'a'
      ? this.comparison.evaluation_a?.['method'] ||
          this.comparison.summary?.evaluation_a_method ||
          'N/A'
      : this.comparison.evaluation_b?.['method'] ||
          this.comparison.summary?.evaluation_b_method ||
          'N/A';
  }

  /**
   * Get evaluation processed items - FIXED to use direct API data
   */
  getEvaluationProcessedItems(evaluation: 'a' | 'b'): number {
    if (!this.comparison) return 0;

    return evaluation === 'a'
      ? this.comparison.evaluation_a?.['processed_items'] || 0
      : this.comparison.evaluation_b?.['processed_items'] || 0;
  }

  /**
   * Get agent name - FIXED to use direct API data
   */
  getAgentName(evaluation: 'a' | 'b'): string {
    if (!this.comparison) return 'N/A';

    const agentId =
      evaluation === 'a'
        ? this.comparison.evaluation_a?.['agent_id']
        : this.comparison.evaluation_b?.['agent_id'];

    return agentId || 'N/A';
  }

  /**
   * Get dataset name - FIXED to use direct API data
   */
  getDatasetName(evaluation: 'a' | 'b'): string {
    if (!this.comparison) return 'N/A';

    const datasetId =
      evaluation === 'a'
        ? this.comparison.evaluation_a?.['dataset_id']
        : this.comparison.evaluation_b?.['dataset_id'];

    return datasetId || 'N/A';
  }

  /**
   * Get overall score - FIXED to use direct API data
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
   * Get statistical power - FIXED to use direct API data
   */
  getStatisticalPower(): string {
    if (this.comparison?.summary?.statistical_power) {
      return (
        this.comparison.summary.statistical_power.power_category || 'Unknown'
      );
    }

    // Fallback based on sample size
    const sampleSize = this.comparison?.summary?.matched_samples || 0;
    if (sampleSize < 5) return 'Very Low';
    if (sampleSize < 10) return 'Low';
    if (sampleSize < 30) return 'Medium';
    return 'High';
  }

  /**
   * Get consistency score - FIXED to use direct API data
   */
  getConsistencyScore(): number | null {
    return this.comparison?.summary?.consistency_score || null;
  }

  /**
   * Get weighted improvement score - FIXED to use direct API data
   */
  getWeightedImprovementScore(): number | null {
    return this.comparison?.summary?.weighted_improvement_score || null;
  }

  /**
   * Get cross method comparison flag - FIXED to use direct API data
   */
  getCrossMethodComparison(): boolean {
    return this.comparison?.summary?.cross_method_comparison || false;
  }

  /**
   * Get top improvements - FIXED to use direct API data
   */
  getTopImprovements(): Array<{
    metric_name: string;
    percentage_change: number;
  }> {
    return this.comparison?.summary?.top_improvements || [];
  }

  /**
   * Get top regressions - FIXED to use direct API data
   */
  getTopRegressions(): Array<{
    metric_name: string;
    percentage_change: number;
  }> {
    return this.comparison?.summary?.top_regressions || [];
  }

  /**
   * Get unchanged metrics count
   */
  getUnchangedMetrics(): number {
    const total = this.comparison?.summary?.total_metrics || 0;
    const improved = this.comparison?.summary?.improved_metrics || 0;
    const regressed = this.comparison?.summary?.regressed_metrics || 0;
    return Math.max(0, total - improved - regressed);
  }

  /**
   * Get metric statistics for detailed view - FIXED to use direct API data
   */
  getMetricStatistics(metricName: string): any {
    return this.comparison?.comparison_results?.['metric_comparison']?.[
      metricName
    ];
  }

  /**
   * Get CSS class for status badge
   */
  getStatusBadgeClass(status: ComparisonStatus): string {
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
        return '';
    }
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
   * Check if comparison has results
   */
  hasResults(): boolean {
    return (
      !!this.comparison?.summary &&
      this.comparison.status === ComparisonStatus.COMPLETED
    );
  }

  /**
   * Check if comparison has sample results
   */
  hasSampleResults(): boolean {
    return this.sampleDifferences.length > 0;
  }

  /**
   * Get formatted overall result - FIXED to use direct API data
   */
  getFormattedOverallResult(): string {
    if (!this.comparison?.summary) return 'N/A';

    if (this.comparison.summary.percentage_change !== undefined) {
      const improvement = this.comparison.summary.percentage_change;
      return `${improvement > 0 ? '+' : ''}${improvement.toFixed(1)}%`;
    }

    return this.comparison.summary.overall_result || 'N/A';
  }

  /**
   * Get result CSS class
   */
  getResultClass(): string {
    if (!this.comparison?.summary) return 'neutral';

    if (this.comparison.summary.percentage_change !== undefined) {
      const improvement = this.comparison.summary.percentage_change;
      if (improvement > 0) return 'improved';
      if (improvement < 0) return 'regressed';
      return 'neutral';
    }

    if (this.comparison.summary.overall_result) {
      if (this.comparison.summary.overall_result === 'improved')
        return 'improved';
      if (this.comparison.summary.overall_result === 'regressed')
        return 'regressed';
    }

    return 'neutral';
  }

  /**
   * Get metric row CSS class
   */
  getMetricRowClass(metric: MetricDifference): string {
    return metric.is_improvement ? 'improved' : 'regressed';
  }

  /**
   * Get metric card CSS class
   */
  getMetricCardClass(metric: MetricDifference): string {
    return metric.is_improvement ? 'improved' : 'regressed';
  }

  /**
   * Get difference CSS class
   */
  getDifferenceClass(metric: MetricDifference): string {
    if (metric.is_improvement) return 'positive';
    if (metric.absolute_difference === 0) return 'neutral';
    return 'negative';
  }

  /**
   * Get formatted difference with sign
   */
  getDifferenceWithSign(metric: MetricDifference): string {
    const sign = metric.is_improvement ? '+' : '';
    return `${sign}${metric.absolute_difference.toFixed(4)}`;
  }

  /**
   * Get formatted percentage difference
   */
  getPercentageDifference(metric: MetricDifference): string {
    const sign = metric.is_improvement ? '+' : '';
    const percentage = metric.percentage_change || 0;
    return `${sign}${percentage.toFixed(1)}%`;
  }

  /**
   * Get impact class
   */
  getImpactClass(metric: MetricDifference): string {
    const percentage = Math.abs(metric.percentage_change || 0);
    if (percentage >= 20) return 'high';
    if (percentage >= 5) return 'medium';
    return 'low';
  }

  /**
   * Get impact label
   */
  getImpactLabel(metric: MetricDifference): string {
    const percentage = Math.abs(metric.percentage_change || 0);
    if (percentage >= 20) return 'High Impact';
    if (percentage >= 5) return 'Medium Impact';
    return 'Low Impact';
  }

  /**
   * Get sample row CSS class
   */
  getSampleRowClass(sample: SampleDifference): string {
    return sample.status === 'improved'
      ? 'improved'
      : sample.status === 'regressed'
      ? 'regressed'
      : '';
  }

  /**
   * Get sample difference CSS class
   */
  getSampleDifferenceClass(sample: SampleDifference | null): string {
    if (!sample) return 'neutral';
    if (sample.status === 'improved') return 'positive';
    if (sample.status === 'regressed') return 'negative';
    return 'neutral';
  }

  /**
   * Get formatted sample difference with sign
   */
  getSampleDifferenceWithSign(sample: SampleDifference | null): string {
    if (!sample || sample.absolute_difference === undefined) return '0.0000';
    const sign = sample.status === 'improved' ? '+' : '';
    return `${sign}${sample.absolute_difference.toFixed(4)}`;
  }

  /**
   * Get formatted sample percentage difference
   */
  getSamplePercentageDifference(sample: SampleDifference | null): string {
    if (!sample || sample.percentage_difference === undefined) return '0.0%';
    const sign = sample.status === 'improved' ? '+' : '';
    return `${sign}${sample.percentage_difference.toFixed(1)}%`;
  }

  /**
   * Get CSS class for sample status badge
   */
  getSampleStatusClass(status: string | undefined): string {
    if (!status) return '';
    switch (status) {
      case 'improved':
        return 'improved';
      case 'regressed':
        return 'regressed';
      case 'unchanged':
        return 'unchanged';
      default:
        return '';
    }
  }

  /**
   * Get config threshold value
   */
  getConfigThreshold(): string {
    return this.comparison?.config?.['threshold']?.toString() || '0.05';
  }

  /**
   * Format JSON data for display
   */
  formatJsonData(data: any): string {
    if (!data) return 'No data available';
    try {
      return JSON.stringify(data, null, 2);
    } catch (e) {
      return String(data);
    }
  }

  // ===== ENHANCED ANALYSIS SUMMARY METHODS =====

  /**
   * Check if there are compatibility warnings
   */
  hasCompatibilityWarnings(): boolean {
    if (!this.comparison) return false;

    // Check if different datasets
    const datasetA = this.comparison.evaluation_a?.['dataset_id'];
    const datasetB = this.comparison.evaluation_b?.['dataset_id'];

    // Check if different sample sizes
    const samplesA = this.getEvaluationProcessedItems('a');
    const samplesB = this.getEvaluationProcessedItems('b');

    return datasetA !== datasetB || samplesA !== samplesB;
  }

  /**
   * Get statistical power CSS class
   */
  getStatisticalPowerClass(): string {
    const power = this.getStatisticalPower().toLowerCase();
    switch (power) {
      case 'high':
        return 'high-power';
      case 'medium':
        return 'medium-power';
      case 'low':
        return 'low-power';
      case 'very low':
        return 'very-low-power';
      default:
        return 'unknown-power';
    }
  }

  /**
   * Get statistical power recommendations
   */
  getStatisticalPowerRecommendations(): string[] {
    return this.comparison?.summary?.statistical_power?.recommendations || [];
  }

  /**
   * Check if there are key findings to display
   */
  hasKeyFindings(): boolean {
    const improvements = this.getTopImprovements();
    const regressions = this.getTopRegressions();
    return (
      improvements.length > 0 ||
      regressions.length > 0 ||
      this.hasAdditionalInsights()
    );
  }

  /**
   * Check if there are additional insights
   */
  hasAdditionalInsights(): boolean {
    return (
      this.getWeightedImprovementScore() !== null ||
      this.getConsistencyScore() !== null ||
      this.getCrossMethodComparison()
    );
  }

  /**
   * Get impact class from percentage change
   */
  getImpactClassFromPercentage(percentage: number): string {
    const absPercentage = Math.abs(percentage);
    if (absPercentage >= 20) return 'high';
    if (absPercentage >= 5) return 'medium';
    return 'low';
  }

  /**
   * Get impact label from percentage change
   */
  getImpactLabelFromPercentage(percentage: number): string {
    const absPercentage = Math.abs(percentage);
    if (absPercentage >= 20) return 'High Impact';
    if (absPercentage >= 5) return 'Medium Impact';
    return 'Low Impact';
  }

  /**
   * Toggle narrative expanded state
   */
  toggleNarrativeExpanded(): void {
    this.isNarrativeExpanded = !this.isNarrativeExpanded;
    this.cdr.markForCheck();
  }

  /**
   * Get recommendations based on analysis
   */
  getRecommendations(): string[] {
    const recommendations: string[] = [];

    // Add statistical power recommendations
    const powerRecs = this.getStatisticalPowerRecommendations();
    recommendations.push(...powerRecs);

    // Add performance-based recommendations
    if (this.comparison?.summary?.overall_result === 'regressed') {
      recommendations.push(
        'Consider investigating the root causes of performance regression'
      );
      recommendations.push(
        'Review configuration differences between evaluations'
      );
    }

    // Add sample size recommendations
    const matchedSamples = this.comparison?.summary?.matched_samples || 0;
    if (matchedSamples < 10) {
      recommendations.push(
        'Increase sample size for more reliable statistical analysis'
      );
    }

    // Add dataset consistency recommendations
    if (this.hasCompatibilityWarnings()) {
      recommendations.push(
        'Use consistent datasets and configurations for fair comparison'
      );
    }

    return recommendations;
  }
}
