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
  Dataset,
  ImpersonationContext,
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

interface MetricsSummary {
  totalMetrics: number;
  passedMetrics: number;
  passRate: number;
  criticalIssues: string[];
  strengths: string[];
  recommendations: string[];
}

interface MetricCategory {
  name: string;
  description: string;
  averageScore: number;
  success: boolean;
  threshold: number;
  samples: { score: number; success: boolean }[];
}

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
  resultsSummary: any = null;
  isLoading = false;
  isLoadingResults = false;
  error: string | null = null;
  resultsError: string | null = null;
  evaluationProgress: EvaluationProgress | null = null;
  evaluationId: string = '';

  // Enhanced metrics data
  metricsCategories: MetricCategory[] = [];
  metricsSummary: MetricsSummary | null = null;

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

  selectedResult: EvaluationResult | null = null;
  showResultDetails = false;
  progressPolling = false;

  // Enums for template access
  EvaluationStatus = EvaluationStatus;
  EvaluationMethod = EvaluationMethod;

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

          this.loadRelatedEntities();

          if (
            evaluation.status === EvaluationStatus.COMPLETED ||
            evaluation.status === EvaluationStatus.FAILED ||
            evaluation.status === EvaluationStatus.CANCELLED
          ) {
            this.loadEvaluationResults();
          }

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
          this.resultsSummary = (response as any).summary || null;
          this.isLoadingResults = false;

          // Update evaluation with impersonation context if available
          if (response.impersonation_context && this.evaluation) {
            this.evaluation.impersonation_context =
              response.impersonation_context;
          }

          // Process metrics data from backend response
          this.processMetricsData();
        },
        error: (error) => {
          this.resultsError = 'Failed to load evaluation results.';
          this.isLoadingResults = false;
          console.error('Error loading evaluation results:', error);
        },
      });
  }

  processMetricsData(): void {
    if (!this.evaluationResults || this.evaluationResults.length === 0) {
      return;
    }

    // Extract unique metrics and calculate averages
    const metricsMap = new Map<string, MetricCategory>();
    let totalSuccessfulMetrics = 0;
    let totalMetrics = 0;

    this.evaluationResults.forEach((result) => {
      if (result.metric_scores && result.metric_scores.length > 0) {
        result.metric_scores.forEach((metric: any) => {
          totalMetrics++;
          if (metric.meta_info?.success) {
            totalSuccessfulMetrics++;
          }

          if (!metricsMap.has(metric.name)) {
            metricsMap.set(metric.name, {
              name: metric.name,
              description: this.getMetricDescription(metric.name),
              averageScore: 0,
              success: false,
              threshold: metric.meta_info?.threshold || 0,
              samples: [],
            });
          }

          const category = metricsMap.get(metric.name)!;
          category.samples.push({
            score: metric.value,
            success: metric.meta_info?.success || false,
          });
        });
      }
    });

    // Calculate averages and overall success
    this.metricsCategories = Array.from(metricsMap.values()).map((category) => {
      const avgScore =
        category.samples.reduce((sum, sample) => sum + sample.score, 0) /
        category.samples.length;
      const successCount = category.samples.filter(
        (sample) => sample.success
      ).length;

      return {
        ...category,
        averageScore: avgScore,
        success: successCount === category.samples.length, // All samples must pass
      };
    });

    // Generate summary insights
    this.generateMetricsSummary();
  }

  generateMetricsSummary(): void {
    if (!this.resultsSummary || !this.metricsCategories.length) return;

    const failedMetrics = this.metricsCategories.filter((m) => !m.success);
    const excellentMetrics = this.metricsCategories.filter(
      (m) => m.success && m.averageScore >= 0.8
    );

    const criticalIssues: string[] = [];
    const strengths: string[] = [];
    const recommendations: string[] = [];

    // Identify critical issues from failed metrics
    failedMetrics.forEach((metric) => {
      criticalIssues.push(
        `${metric.name} underperforming (${(metric.averageScore * 100).toFixed(
          1
        )}% avg)`
      );
    });

    // Identify strengths from excellent metrics
    excellentMetrics.forEach((metric) => {
      strengths.push(
        `Excellent ${metric.name} performance (${(
          metric.averageScore * 100
        ).toFixed(1)}% avg)`
      );
    });

    // Generate recommendations based on performance
    if (this.resultsSummary.pass_rate < 70) {
      recommendations.push('Review evaluation strategy - pass rate below 70%');
    }

    failedMetrics.forEach((metric) => {
      if (metric.name.toLowerCase().includes('relevancy')) {
        recommendations.push(
          'Optimize context retrieval and ranking algorithms'
        );
      } else if (metric.name.toLowerCase().includes('correctness')) {
        recommendations.push('Review model accuracy and training data quality');
      }
    });

    this.metricsSummary = {
      totalMetrics: this.metricsCategories.length,
      passedMetrics: this.metricsCategories.filter((m) => m.success).length,
      passRate: this.resultsSummary.pass_rate,
      criticalIssues,
      strengths,
      recommendations: recommendations.length
        ? recommendations
        : ['Continue current evaluation strategy'],
    };
  }

  getMetricDescription(metricName: string): string {
    const descriptions: { [key: string]: string } = {
      'Answer Relevancy': 'Response relevance to query',
      'Correctness (GEval)': 'Factual accuracy score',
      'Completeness (GEval)': 'Response completeness',
      Faithfulness: 'Context consistency',
      'Contextual Precision': 'Context ranking quality',
      'Contextual Relevancy': 'Context relevance score',
      'Contextual Recall': 'Context completeness',
      Hallucination: 'Factual inconsistencies',
      Toxicity: 'Harmful content detection',
      Bias: 'Unfair bias detection',
    };
    return descriptions[metricName] || 'Evaluation metric';
  }

  getMetricStatusClass(metric: MetricCategory | any): string {
    // Handle MetricCategory objects
    if (metric.success !== undefined) {
      if (metric.success) {
        return (metric.averageScore || metric.value || 0) >= 0.8
          ? 'excellent'
          : 'good';
      }
      return (metric.averageScore || metric.value || 0) >= 0.5
        ? 'warning'
        : 'danger';
    }

    // Handle metric objects with meta_info
    if (metric.meta_info?.success !== undefined) {
      return metric.meta_info.success ? 'pass' : 'fail';
    }

    // Fallback for other cases
    const value = metric.averageScore || metric.value || 0;
    if (value >= 0.8) return 'excellent';
    if (value >= 0.6) return 'good';
    if (value >= 0.5) return 'warning';
    return 'danger';
  }

  getMetricStatusText(metric: MetricCategory): string {
    if (metric.success) {
      return metric.averageScore >= 0.8 ? 'EXCELLENT' : 'GOOD';
    }
    return metric.averageScore >= 0.5 ? 'FAIR' : 'POOR';
  }

  loadRelatedEntities(): void {
    if (!this.evaluation) return;

    if (this.evaluation.agent_id && !this.evaluation.agent) {
      this.agentService
        .getAgent(this.evaluation.agent_id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (agent) => {
            this.evaluation!.agent = agent;
          },
          error: (error) => {
            console.error('Error loading agent details:', error);
          },
        });
    }

    if (this.evaluation.dataset_id && !this.evaluation.dataset) {
      this.datasetService
        .getDataset(this.evaluation.dataset_id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (datasetDetailResponse) => {
            this.evaluation!.dataset = datasetDetailResponse.dataset;
          },
          error: (error) => {
            console.error('Error loading dataset details:', error);
          },
        });
    }

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
          },
          error: (error) => {
            console.error('Error loading prompt details:', error);
          },
        });
    }
  }

  editEvaluation(): void {
    if (!this.evaluation) return;

    if (this.evaluation.status !== EvaluationStatus.PENDING) {
      this.notificationService.error(
        `Cannot edit ${this.evaluation.status.toLowerCase()} evaluation`
      );
      return;
    }

    this.router.navigate(['app/evaluations', this.evaluation.id, 'edit']);
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

  navigateToCreateReport(): void {
    if (!this.evaluation) return;

    if (this.evaluation.status !== EvaluationStatus.COMPLETED) {
      this.notificationService.error(
        'Reports can only be generated for completed evaluations'
      );
      return;
    }

    this.router.navigate(['app/reports/create'], {
      queryParams: { evaluation_id: this.evaluation.id },
    });
  }

  viewResultDetails(result: any): void {
    this.selectedResult = result;
    this.showResultDetails = true;
    this.activeResultTab = 'query';
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

  getQueryText(result: any): string {
    if (!result || !result.input_data) return 'No query available';
    return (
      result.input_data.input ||
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
      result.output_data.actual_output ||
      result.output_data.response ||
      result.output_data.answer ||
      result.output_data.generated_text ||
      JSON.stringify(result.output_data)
    );
  }

  getExpectedOutputText(result: any): string {
    if (!result || !result.input_data) return 'No expected output available';
    return result.input_data.expected_output || 'No expected output provided';
  }

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

  getSuccessCount(samples: { success: boolean }[]): number {
    return samples ? samples.filter((sample) => sample.success).length : 0;
  }

  getOverallScoreClass(score: number | undefined): { [key: string]: boolean } {
    if (!score) return {};
    return {
      excellent: score >= 0.8,
      good: score >= 0.6 && score < 0.8,
      poor: score < 0.6,
    };
  }

  getMetricStatus(metric: any): string {
    return metric.meta_info?.success ? 'PASS' : 'FAIL';
  }

  getMetricThreshold(metric: any): number | null {
    return metric.meta_info?.threshold || null;
  }

  getMetricReason(metric: any): string | null {
    return metric.meta_info?.reason || null;
  }

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

  getJsonString(obj: any): string {
    try {
      return JSON.stringify(obj, null, 2);
    } catch (error) {
      console.error('Error converting to JSON:', error);
      return String(obj);
    }
  }

  exportResultDetails(): void {
    if (!this.selectedResult) return;

    const exportData = {
      evaluation_id: this.evaluation?.id,
      result_id: this.selectedResult.id,
      query: this.getQueryText(this.selectedResult),
      context: this.getContextText(this.selectedResult),
      output: this.getOutputText(this.selectedResult),
      expected_output: this.getExpectedOutputText(this.selectedResult),
      metrics: this.selectedResult.metric_scores,
      overall_score: this.selectedResult.overall_score,
      passed: this.selectedResult['passed'],
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

  /**
   * Check if evaluation has impersonation context
   */
  hasImpersonationContext(): boolean {
    return !!this.evaluation?.impersonation_context?.is_impersonated;
  }

  /**
   * Get impersonated user display text
   */
  getImpersonatedUserDisplay(): string {
    if (!this.evaluation?.impersonation_context?.is_impersonated) {
      return '';
    }

    const context = this.evaluation.impersonation_context;

    // Use display name if available, otherwise construct from user info
    if (context.impersonated_user_display) {
      return context.impersonated_user_display;
    }

    if (context.impersonated_user_info) {
      const userInfo = context.impersonated_user_info;
      return (
        userInfo.name || userInfo.preferred_username || userInfo.employee_id
      );
    }

    return context.impersonated_user_id || 'Unknown User';
  }

  /**
   * Get impersonated user email
   */
  getImpersonatedUserEmail(): string {
    return (
      this.evaluation?.impersonation_context?.impersonated_user_info?.email ||
      ''
    );
  }

  /**
   * Get impersonated user ID
   */
  getImpersonatedUserId(): string {
    return this.evaluation?.impersonation_context?.impersonated_user_id || '';
  }
}
