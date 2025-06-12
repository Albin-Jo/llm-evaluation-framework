import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, interval, takeUntil, forkJoin } from 'rxjs';
import { switchMap, filter } from 'rxjs/operators';
import {
  EvaluationDetail,
  EvaluationMethod,
  EvaluationProgress,
  EvaluationResult,
  EvaluationStatus,
  MetricScore,
  Prompt,
  Dataset, // Make sure Dataset is imported
} from '@ngtx-apps/data-access/models';
import {
  EvaluationService,
  AgentService,
  DatasetService,
  PromptService,
} from '@ngtx-apps/data-access/services';
import {
  QracButtonComponent,
  QracTagButtonComponent,
} from '@ngtx-apps/ui/components';
import {
  ConfirmationDialogService,
  NotificationService,
} from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-evaluation-detail',
  standalone: true,
  imports: [CommonModule, RouterModule],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './evaluation-detail.page.html',
  styleUrls: ['./evaluation-detail.page.scss'],
})
export class EvaluationDetailPage implements OnInit, OnDestroy {
  evaluation: EvaluationDetail | null = null;
  evaluationResults: any[] = [];
  isLoading = false;
  isLoadingResults = false;
  error: string | null = null;
  resultsError: string | null = null;
  evaluationProgress: EvaluationProgress | null = null;
  evaluationId: string = '';

  // Pagination for results
  resultsPage = 1;
  resultsLimit = 100;
  totalResults = 0;

  // Enhanced modal state
  activeResultTab: 'query' | 'context' | 'output' | 'metrics' = 'query';
  expandedSections = {
    query: false,
    context: false,
    output: false,
    input: false,
    raw: false,
  };

  // For displaying result details
  selectedResult: EvaluationResult | null = null;
  showResultDetails = false;

  // Polling for progress updates
  progressPolling = false;

  // Enums for template access
  EvaluationStatus = EvaluationStatus;
  EvaluationMethod = EvaluationMethod;

  // Metrics charting
  metricsData: { name: string; value: number }[] = [];

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private evaluationService: EvaluationService,
    private agentService: AgentService,
    private datasetService: DatasetService,
    private promptService: PromptService,
    private confirmationDialogService: ConfirmationDialogService,
    private notificationService: NotificationService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.pipe(takeUntil(this.destroy$)).subscribe((params) => {
      const id = params.get('id');
      if (id) {
        this.evaluationId = id;
        this.loadEvaluationData(id);
      } else {
        this.error = 'Evaluation ID not found.';
      }
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.stopProgressPolling();
  }

  // ISSUE 4 FIX: Add status validation before editing
  editEvaluation(): void {
    if (!this.evaluation) return;

    // Check if evaluation can be edited (only PENDING status)
    if (this.evaluation.status !== EvaluationStatus.PENDING) {
      this.notificationService.error(
        `Cannot edit ${this.evaluation.status.toLowerCase()} evaluation`
      );
      return;
    }

    this.router.navigate(['app/evaluations', this.evaluation.id, 'edit']);
  }

  /**
   * Load both evaluation details and results
   */
  loadEvaluationData(id: string): void {
    this.isLoading = true;
    this.error = null;

    this.evaluationService
      .getEvaluation(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (evaluation) => {
          this.evaluation = evaluation;
          this.isLoading = false;

          // Load related entities
          this.loadRelatedEntities();

          // Load results if evaluation has completed
          if (
            evaluation.status === EvaluationStatus.COMPLETED ||
            evaluation.status === EvaluationStatus.FAILED ||
            evaluation.status === EvaluationStatus.CANCELLED
          ) {
            this.loadEvaluationResults();
          }

          // Handle progress polling
          if (evaluation.status === EvaluationStatus.RUNNING) {
            this.startProgressPolling();
          } else if (evaluation.status === EvaluationStatus.PENDING) {
            this.loadEvaluationProgress();
          }
        },
        error: (error) => {
          this.error = 'Failed to load evaluation details. Please try again.';
          this.isLoading = false;
          console.error('Error loading evaluation:', error);
        },
      });
  }

  /**
   * Load evaluation results separately
   */
  loadEvaluationResults(): void {
    if (!this.evaluation) return;

    this.isLoadingResults = true;
    this.resultsError = null;

    this.evaluationService
      .getEvaluationResults(this.evaluation.id, 0, 100)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.evaluationResults = response.items || [];
          this.totalResults = response.total || 0;
          this.isLoadingResults = false;
          this.prepareMetricsData();
        },
        error: (error) => {
          this.resultsError = 'Failed to load evaluation results.';
          this.isLoadingResults = false;
          console.error('Error loading evaluation results:', error);
        },
      });
  }

  loadRelatedEntities(): void {
    if (!this.evaluation) return;

    // Load agent details if we have agent_id
    if (this.evaluation.agent_id && !this.evaluation.agent) {
      this.agentService
        .getAgent(this.evaluation.agent_id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (agent) => {
            this.evaluation!.agent = agent;
            console.log('✅ Loaded agent:', agent.name); // Debug log
          },
          error: (error) => {
            console.error('❌ Error loading agent details:', error);
          },
        });
    }

    // FIXED: Load dataset details - extract dataset from DatasetDetailResponse
    if (this.evaluation.dataset_id && !this.evaluation.dataset) {
      this.datasetService
        .getDataset(this.evaluation.dataset_id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (datasetDetailResponse) => {
            // DatasetService.getDataset() returns { dataset: Dataset, documents: Document[] }
            // We need to extract the dataset property
            this.evaluation!.dataset = datasetDetailResponse.dataset;
            console.log(
              '✅ Loaded dataset:',
              datasetDetailResponse.dataset.name
            ); // Debug log
          },
          error: (error) => {
            console.error('❌ Error loading dataset details:', error);
          },
        });
    }

    // Load prompt details if we have prompt_id
    if (this.evaluation.prompt_id && !this.evaluation.prompt) {
      this.promptService
        .getPromptById(this.evaluation.prompt_id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (prompt) => {
            this.evaluation!.prompt = {
              id: prompt.id,
              name: prompt.name,
              description: prompt.description,
              template: prompt.content || '',
              created_at: prompt.created_at,
              updated_at: prompt.updated_at,
              version: prompt.version ? Number(prompt.version) : undefined,
            } as Prompt;
            console.log('✅ Loaded prompt:', prompt.name); // Debug log
          },
          error: (error) => {
            console.error('❌ Error loading prompt details:', error);
          },
        });
    }
  }

  startEvaluation(): void {
    if (!this.evaluation) return;

    this.confirmationDialogService
      .confirm({
        title: 'Start Evaluation',
        message: 'Are you sure you want to start this evaluation?',
        confirmText: 'Start',
        cancelText: 'Cancel',
        type: 'info',
      })
      .subscribe((confirmed) => {
        if (confirmed) {
          this.evaluationService
            .startEvaluation(this.evaluation!.id)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: (evaluation) => {
                this.notificationService.success(
                  'Evaluation started successfully'
                );
                this.loadEvaluationData(this.evaluation!.id);
                this.startProgressPolling();
              },
              error: (error) => {
                this.notificationService.error('Failed to start evaluation');
                console.error('Error starting evaluation:', error);
              },
            });
        }
      });
  }

  cancelEvaluation(): void {
    if (!this.evaluation) return;

    this.confirmationDialogService
      .confirm({
        title: 'Cancel Evaluation',
        message: 'Are you sure you want to cancel this evaluation?',
        confirmText: 'Cancel Evaluation',
        cancelText: 'Keep Running',
        type: 'warning',
      })
      .subscribe((confirmed) => {
        if (confirmed) {
          this.evaluationService
            .cancelEvaluation(this.evaluation!.id)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: (evaluation) => {
                this.notificationService.success(
                  'Evaluation cancelled successfully'
                );
                this.loadEvaluationData(this.evaluation!.id);
                this.stopProgressPolling();
              },
              error: (error) => {
                this.notificationService.error('Failed to cancel evaluation');
                console.error('Error cancelling evaluation:', error);
              },
            });
        }
      });
  }

  deleteEvaluation(): void {
    if (!this.evaluation) return;

    this.confirmationDialogService
      .confirmDelete('Evaluation')
      .subscribe((confirmed) => {
        if (confirmed) {
          this.evaluationService
            .deleteEvaluation(this.evaluation!.id)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.notificationService.success(
                  'Evaluation deleted successfully'
                );
                this.router.navigate(['app/evaluations']);
              },
              error: (error) => {
                this.notificationService.error('Failed to delete evaluation');
                console.error('Error deleting evaluation:', error);
              },
            });
        }
      });
  }

  // ISSUE 1: Enhanced result details methods
  viewResultDetails(result: any): void {
    this.selectedResult = result;
    this.showResultDetails = true;
    this.activeResultTab = 'query'; // Reset to first tab
    // Reset expanded sections
    this.expandedSections = {
      query: false,
      context: false,
      output: false,
      input: false,
      raw: false,
    };
  }

  closeResultDetails(): void {
    this.showResultDetails = false;
    this.selectedResult = null;
  }

  setActiveResultTab(tab: 'query' | 'context' | 'output' | 'metrics'): void {
    this.activeResultTab = tab;
  }
  // Text extraction methods for different result fields
  getQueryText(result: any): string {
    if (!result || !result.input_data) return 'No query available';

    // Handle different possible query field names
    return (
      result.input_data.query ||
      result.input_data.question ||
      result.input_data.user_input ||
      JSON.stringify(result.input_data)
    );
  }

  getContextText(result: any): string {
    if (!result || !result.input_data) return 'No context available';

    const context =
      result.input_data.context ||
      result.input_data.contexts ||
      result.input_data.retrieved_context;

    if (Array.isArray(context)) {
      return context.join('\n\n');
    } else if (typeof context === 'string') {
      return context;
    }

    return 'No context available';
  }

  getContextArray(result: any): any[] {
    if (!result || !result.input_data) return [];

    const context =
      result.input_data.context ||
      result.input_data.contexts ||
      result.input_data.retrieved_context;

    if (Array.isArray(context)) {
      return context;
    } else if (typeof context === 'string') {
      return [{ content: context }];
    }

    return [];
  }

  getOutputText(result: any): string {
    if (!result || !result.output_data) return 'No output available';

    return (
      result.output_data.response ||
      result.output_data.answer ||
      result.output_data.generated_text ||
      JSON.stringify(result.output_data)
    );
  }

  // Utility methods for text management
  shouldShowExpandButton(text: string | undefined): boolean {
    return !!(text && typeof text === 'string' && text.length > 300);
  }

  getCharacterCountText(text: string | undefined, isExpanded: boolean): string {
    if (!text || typeof text !== 'string') return '';

    if (isExpanded) {
      return `${text.length} characters`;
    } else {
      const visibleChars = Math.min(300, text.length);
      return `${visibleChars} of ${text.length} characters`;
    }
  }

  getMetricStatus(metric: any): string {
    // Use backend-calculated success if available
    if (metric.meta_info && typeof metric.meta_info.success === 'boolean') {
      return metric.meta_info.success ? 'PASS' : 'FAIL';
    }
    
    // Fallback to frontend calculation (should not be needed anymore)
    if (metric.value >= 0.7) return 'PASS';
    if (metric.value >= 0.5) return 'WARN';
    return 'FAIL';
  }
  getMetricStatusClass(metric: any): string {
    // Use backend-calculated success if available
    if (metric.meta_info && typeof metric.meta_info.success === 'boolean') {
      return metric.meta_info.success ? 'pass' : 'fail';
    }
    
    // Fallback to frontend calculation
    if (metric.value >= 0.7) return 'pass';
    if (metric.value >= 0.5) return 'warn';
    return 'fail';
  }
  
  getMetricThreshold(metric: any): number | null {
    return metric.meta_info && metric.meta_info.threshold ? metric.meta_info.threshold : null;
  }
  
  getMetricReason(metric: any): string | null {
    return metric.meta_info && metric.meta_info.reason ? metric.meta_info.reason : null;
  }

  // Copy to clipboard functionality
  copyToClipboard(text: string): void {
    navigator.clipboard.writeText(text).then(
      () => {
        this.notificationService.success('Copied to clipboard');
      },
      (err) => {
        console.error('Could not copy text: ', err);
        this.notificationService.error('Failed to copy text');
      }
    );
  }

  // Helper method to safely convert objects to JSON strings
  getJsonString(obj: any): string {
    try {
      return JSON.stringify(obj, null, 2);
    } catch (error) {
      console.error('Error converting to JSON:', error);
      return String(obj);
    }
  }

  // ISSUE 2 FIX: Generate Report Button
  navigateToCreateReport(): void {
    if (!this.evaluation) return;

    // Check if evaluation is completed
    if (this.evaluation.status !== EvaluationStatus.COMPLETED) {
      this.notificationService.error(
        'Reports can only be generated for completed evaluations'
      );
      return;
    }

    // Navigate to report creation with evaluation ID
    this.router.navigate(['app/reports/create'], {
      queryParams: { evaluation_id: this.evaluation.id },
    });
  }

  exportResultDetails(): void {
    if (!this.selectedResult) return;

    const exportData = {
      evaluation_id: this.evaluation?.id,
      result_id: this.selectedResult.id,
      query: this.getQueryText(this.selectedResult),
      context: this.getContextText(this.selectedResult),
      output: this.getOutputText(this.selectedResult),
      metrics: this.selectedResult.metric_scores,
      overall_score: this.selectedResult.overall_score,
      timestamp: new Date().toISOString(),
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri =
      'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

    const exportFileDefaultName = `evaluation_result_${this.selectedResult.id}.json`;

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();

    this.notificationService.success('Result exported successfully');
  }

  loadEvaluationProgress(): void {
    if (!this.evaluation) return;

    this.evaluationService
      .getEvaluationProgress(this.evaluation.id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (progress) => {
          this.evaluationProgress = this.transformProgressResponse(progress);
        },
        error: (error) => {
          console.error('Error loading evaluation progress:', error);
        },
      });
  }

  private transformProgressResponse(backendProgress: any): EvaluationProgress {
    return {
      total: backendProgress.total_items || 0,
      completed: backendProgress.completed_items || 0,
      failed: 0,
      percentage: backendProgress.progress_percentage || 0,
      percentage_complete: backendProgress.progress_percentage || 0,
      processed_items: backendProgress.completed_items || 0,
      total_items: backendProgress.total_items || 0,
      estimated_completion: this.calculateEstimatedCompletion(
        backendProgress.estimated_time_remaining_seconds
      ),
      eta_seconds: backendProgress.estimated_time_remaining_seconds,
      status: backendProgress.status as EvaluationStatus,
    };
  }

  private calculateEstimatedCompletion(
    etaSeconds?: number
  ): string | undefined {
    if (!etaSeconds || etaSeconds <= 0) return undefined;

    const now = new Date();
    const estimatedTime = new Date(now.getTime() + etaSeconds * 1000);
    return estimatedTime.toISOString();
  }

  startProgressPolling(): void {
    this.progressPolling = true;

    interval(15000)
      .pipe(
        takeUntil(this.destroy$),
        filter(() => this.progressPolling),
        switchMap(() =>
          this.evaluationService.getEvaluationProgress(this.evaluationId)
        )
      )
      .subscribe({
        next: (progress) => {
          this.evaluationProgress = this.transformProgressResponse(progress);

          if (this.evaluationProgress.status !== EvaluationStatus.RUNNING) {
            this.loadEvaluationData(this.evaluationId);
            this.stopProgressPolling();
          }
        },
        error: (error) => {
          console.error('Error polling evaluation progress:', error);
        },
      });
  }

  stopProgressPolling(): void {
    this.progressPolling = false;
  }

  prepareMetricsData(): void {
    if (!this.evaluationResults || this.evaluationResults.length === 0) {
      return;
    }

    const metricNames = new Set<string>();
    const metricValues: Record<string, number[]> = {};

    this.evaluationResults.forEach((result) => {
      if (result['metric_scores'] && result['metric_scores'].length > 0) {
        result['metric_scores'].forEach((metric: any) => {
          metricNames.add(metric.name);
          if (!metricValues[metric.name]) {
            metricValues[metric.name] = [];
          }
          metricValues[metric.name].push(metric.value);
        });
      }
    });

    this.metricsData = Array.from(metricNames).map((name) => {
      const values = metricValues[name];
      const average =
        values.reduce((sum, value) => sum + value, 0) / values.length;
      return {
        name,
        value: parseFloat(average.toFixed(2)),
      };
    });

    this.metricsData.sort((a, b) => b.value - a.value);
  }

  getStatusBadgeClass(status: EvaluationStatus): string {
    switch (status) {
      case EvaluationStatus.COMPLETED:
        return 'completed';
      case EvaluationStatus.RUNNING:
        return 'running';
      case EvaluationStatus.PENDING:
        return 'pending';
      case EvaluationStatus.FAILED:
        return 'failed';
      case EvaluationStatus.CANCELLED:
        return 'cancelled';
      default:
        return '';
    }
  }

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

  formatDuration(
    startTime: string | undefined,
    endTime: string | undefined
  ): string {
    if (!startTime || !endTime) return 'N/A';

    try {
      const start = new Date(startTime).getTime();
      const end = new Date(endTime).getTime();
      const durationMs = end - start;

      if (durationMs < 0) return 'Invalid duration';

      const seconds = Math.floor(durationMs / 1000);
      if (seconds < 60) return `${seconds} seconds`;

      const minutes = Math.floor(seconds / 60);
      if (minutes < 60) return `${minutes} min ${seconds % 60} sec`;

      const hours = Math.floor(minutes / 60);
      return `${hours} hr ${minutes % 60} min`;
    } catch (e) {
      return 'Invalid duration';
    }
  }

  canStartEvaluation(): boolean {
    return this.evaluation?.status === EvaluationStatus.PENDING;
  }

  canCancelEvaluation(): boolean {
    return this.evaluation?.status === EvaluationStatus.RUNNING;
  }

  hasResults(): boolean {
    return this.evaluationResults && this.evaluationResults.length > 0;
  }

  viewAgent(): void {
    if (this.evaluation?.agent_id) {
      this.router.navigate(['app/agents', this.evaluation.agent_id]);
    }
  }

  viewDataset(): void {
    if (this.evaluation?.dataset_id) {
      this.router.navigate([
        'app/datasets/datasets',
        this.evaluation.dataset_id,
      ]);
    }
  }

  viewPrompt(): void {
    if (this.evaluation?.prompt_id) {
      this.router.navigate(['app/prompts', this.evaluation.prompt_id]);
    }
  }

  toggleSection(section: string): void {
    if (section in this.expandedSections) {
      this.expandedSections[section as keyof typeof this.expandedSections] =
        !this.expandedSections[section as keyof typeof this.expandedSections];
    }
  }

  refreshResults(): void {
    this.loadEvaluationResults();
  }

  isResultsLoading(): boolean {
    return this.isLoadingResults;
  }

  getDisplayResults(): any[] {
    return this.evaluationResults;
  }
}
