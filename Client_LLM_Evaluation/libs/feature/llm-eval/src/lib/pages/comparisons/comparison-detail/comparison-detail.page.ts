import {
  Component,
  OnInit,
  OnDestroy,
  ElementRef,
  NO_ERRORS_SCHEMA,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, takeUntil, forkJoin, of } from 'rxjs';
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

  // Metrics data
  metricDifferences: MetricDifference[] = [];

  // Visualization data
  selectedVisualization: 'radar' | 'bar' | 'line' = 'radar';
  visualizationData: VisualizationData | null = null;

  // Sample data
  sampleDifferences: SampleDifference[] = [];
  filteredSamples: SampleDifference[] = [];
  sampleFilter: string = 'all';
  sampleSort: string = 'difference';

  // Sample details modal
  showSampleDetails = false;
  selectedSample: SampleDifference | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private comparisonService: ComparisonService,
    private evaluationService: EvaluationService,
    private confirmationDialogService: ConfirmationDialogService,
    private notificationService: NotificationService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.pipe(takeUntil(this.destroy$)).subscribe((params) => {
      const id = params.get('id');
      if (id) {
        this.comparisonId = id;
        this.loadComparison(id);
      } else {
        this.error = 'Comparison ID not found.';
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

    this.comparisonService
      .getComparison(id)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isLoading = false;
        })
      )
      .subscribe({
        next: (comparison) => {
          this.comparison = comparison;

          // Process metrics data if available
          if (
            comparison.metric_differences &&
            comparison.metric_differences.length > 0
          ) {
            this.metricDifferences = comparison.metric_differences;
          } else {
            // If no metrics in the response, try to load them separately
            this.loadMetrics();
          }

          // Process sample differences if available
          if (comparison.result_differences) {
            this.processSampleDifferences(comparison.result_differences);
          }

          // Load visualization data for the default visualization
          this.loadVisualizationData(this.selectedVisualization);
        },
        error: (error) => {
          this.error = 'Failed to load comparison details. Please try again.';
          console.error('Error loading comparison:', error);
        },
      });
  }

  /**
   * Load metrics data separately
   */
  loadMetrics(): void {
    if (!this.comparison || !this.comparisonId) return;

    this.comparisonService
      .getComparisonMetrics(this.comparisonId)
      .pipe(
        takeUntil(this.destroy$),
        catchError((error) => {
          console.error('Error loading metrics:', error);
          return of([]);
        })
      )
      .subscribe((metrics) => {
        this.metricDifferences = metrics;
      });
  }

  /**
   * Process sample differences from the response
   */
  processSampleDifferences(
    resultDifferences: Record<string, SampleDifference[]>
  ): void {
    // Flatten the samples from the result differences
    this.sampleDifferences = Object.values(resultDifferences)
      .flat()
      .filter((sample) => !!sample); // Filter out any null/undefined values

    // Apply initial filtering and sorting
    this.filterSamples();
  }

  /**
   * Load visualization data
   */
  loadVisualizationData(type: 'radar' | 'bar' | 'line'): void {
    if (!this.comparison || !this.comparisonId || !this.hasResults()) return;

    this.comparisonService
      .getVisualizationData(this.comparisonId, type)
      .pipe(
        takeUntil(this.destroy$),
        catchError((error) => {
          console.error(`Error loading ${type} visualization:`, error);
          return of(null);
        })
      )
      .subscribe((data) => {
        if (data) {
          this.visualizationData = data;
        }
      });
  }

  /**
   * Filter and sort samples based on user selection
   */
  filterSamples(): void {
    if (!this.sampleDifferences || this.sampleDifferences.length === 0) {
      this.filteredSamples = [];
      return;
    }

    // Apply filter
    let filtered = [...this.sampleDifferences];

    if (this.sampleFilter !== 'all') {
      filtered = filtered.filter(
        (sample) => sample.status === this.sampleFilter
      );
    }

    // Apply sort
    switch (this.sampleSort) {
      case 'difference':
        filtered.sort((a, b) => {
          const diffA = Math.abs(a.absolute_difference || 0);
          const diffB = Math.abs(b.absolute_difference || 0);
          return diffB - diffA; // Sort by absolute difference (largest first)
        });
        break;
      case 'id':
        filtered.sort((a, b) => {
          return a.sample_id.localeCompare(b.sample_id);
        });
        break;
      case 'score_a':
        filtered.sort((a, b) => {
          return (b.evaluation_a_score || 0) - (a.evaluation_a_score || 0);
        });
        break;
      case 'score_b':
        filtered.sort((a, b) => {
          return (b.evaluation_b_score || 0) - (a.evaluation_b_score || 0);
        });
        break;
    }

    this.filteredSamples = filtered;
  }

  /**
   * Sort samples based on user selection
   */
  sortSamples(): void {
    this.filterSamples(); // Just reapply filtering which includes sorting
  }

  /**
   * Handle visualization type selection
   */
  selectVisualization(type: 'radar' | 'bar' | 'line'): void {
    this.selectedVisualization = type;
    this.loadVisualizationData(type);
  }

  /**
   * Show detailed view of a sample
   */
  viewSampleDetails(sample: SampleDifference): void {
    this.selectedSample = sample;
    this.showSampleDetails = true;
  }

  /**
   * Close the sample details modal
   */
  closeSampleDetails(): void {
    this.showSampleDetails = false;
    this.selectedSample = null;
  }

  /**
   * Run the comparison calculation
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

          this.comparisonService
            .runComparison(this.comparisonId)
            .pipe(
              takeUntil(this.destroy$),
              finalize(() => (this.isLoading = false))
            )
            .subscribe({
              next: () => {
                this.notificationService.success('Comparison is running');
                this.loadComparison(this.comparisonId);
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
   * Edit the comparison
   */
  editComparison(): void {
    if (this.comparison) {
      this.router.navigate(['app/comparisons', this.comparison.id, 'edit']);
    }
  }

  /**
   * Delete the comparison
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
   * Navigate to a specific evaluation
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
   * Check if comparison can be run (only PENDING or FAILED comparisons)
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
   * Get formatted overall result
   */
  getFormattedOverallResult(): string {
    if (!this.comparison?.summary?.overall_result) return 'N/A';

    const improvement = this.comparison.summary.overall_result;
    return `${improvement > 0 ? '+' : ''}${(improvement * 100).toFixed(
      1
    )}% Improvement`;
  }

  /**
   * Get result CSS class
   */
  getResultClass(): string {
    if (!this.comparison?.summary?.overall_result) return 'neutral';

    const improvement = this.comparison.summary.overall_result;

    if (improvement > 0) {
      return 'improved';
    } else if (improvement < 0) {
      return 'regressed';
    } else {
      return 'neutral';
    }
  }

  /**
   * Get improved metrics percentage
   */
  getImprovedPercentage(): string {
    if (!this.comparison?.summary) return '0%';

    const improved = this.comparison.summary.improved_metrics || 0;
    const total = this.comparison.summary.total_metrics || 1; // Avoid division by zero

    return `${Math.round((improved / total) * 100)}%`;
  }

  /**
   * Get regressed metrics percentage
   */
  getRegressedPercentage(): string {
    if (!this.comparison?.summary) return '0%';

    const regressed = this.comparison.summary.regressed_metrics || 0;
    const total = this.comparison.summary.total_metrics || 1; // Avoid division by zero

    return `${Math.round((regressed / total) * 100)}%`;
  }

  /**
   * Get metric difference CSS class
   */
  getDifferenceClass(metric: MetricDifference): string {
    if (metric.is_improvement) {
      return 'positive';
    } else if (metric.absolute_difference === 0) {
      return 'neutral';
    } else {
      return 'negative';
    }
  }

  /**
   * Get formatted difference with sign
   */
  getDifferenceWithSign(metric: MetricDifference): string {
    const sign = metric.is_improvement ? '+' : '';
    return `${sign}${metric.absolute_difference.toFixed(2)}`;
  }

  /**
   * Get formatted percentage difference
   */
  getPercentageDifference(metric: MetricDifference): string {
    const sign = metric.is_improvement ? '+' : '';
    return `${sign}${(metric.percentage_difference * 100).toFixed(1)}%`;
  }

  /**
   * Get sample difference CSS class
   */
  getSampleDifferenceClass(sample: SampleDifference | null): string {
    if (!sample || !sample.absolute_difference) return 'neutral';

    if (sample.status === 'improved') {
      return 'positive';
    } else if (sample.status === 'regressed') {
      return 'negative';
    } else {
      return 'neutral';
    }
  }

  /**
   * Get formatted sample difference with sign
   */
  getSampleDifferenceWithSign(sample: SampleDifference | null): string {
    if (!sample || !sample.absolute_difference) return '0.00';

    const sign = sample.status === 'improved' ? '+' : '';
    return `${sign}${sample.absolute_difference.toFixed(2)}`;
  }

  /**
   * Get formatted sample percentage difference
   */
  getSamplePercentageDifference(sample: SampleDifference | null): string {
    if (!sample || !sample.percentage_difference) return '0.0%';

    const sign = sample.status === 'improved' ? '+' : '';
    return `${sign}${(sample.percentage_difference * 100).toFixed(1)}%`;
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
   * Get evaluation dataset name
   */
  getEvaluationDatasetName(): string {
    if (!this.comparison) return '';

    if (this.comparison.evaluation_a?.['dataset']?.name) {
      return this.comparison.evaluation_a['dataset'].name;
    }

    return 'N/A';
  }

  /**
   * Get evaluation method
   */
  getEvaluationMethod(): string {
    if (!this.comparison) return '';

    if (this.comparison.evaluation_a?.['method']) {
      return this.comparison.evaluation_a['method'];
    }

    return 'N/A';
  }

  /**
   * Get config threshold value
   */
  getConfigThreshold(): string {
    if (!this.comparison || !this.comparison.config) return '0.05';

    if (this.comparison.config['threshold']) {
      return this.comparison.config['threshold'].toString();
    }

    return '0.05';
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
}
