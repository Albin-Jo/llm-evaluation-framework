/* Path: libs/feature/llm-eval/src/lib/pages/evaluations/evaluation-detail/evaluation-detail.page.ts (Update) */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, interval, takeUntil } from 'rxjs';
import { switchMap, filter } from 'rxjs/operators';
import {
  EvaluationDetail,
  EvaluationMethod,
  EvaluationProgress,
  EvaluationResult,
  EvaluationStatus,
  MetricScore,
  Prompt,
  Dataset
} from '@ngtx-apps/data-access/models';
import { EvaluationService, AgentService, DatasetService, PromptService } from '@ngtx-apps/data-access/services';
import {
  QracButtonComponent,
  QracTagButtonComponent
} from '@ngtx-apps/ui/components';
import {
  ConfirmationDialogService,
  NotificationService
} from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-evaluation-detail',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './evaluation-detail.page.html',
  styleUrls: ['./evaluation-detail.page.scss']
})
export class EvaluationDetailPage implements OnInit, OnDestroy {
  evaluation: EvaluationDetail | null = null;
  isLoading = false;
  error: string | null = null;
  evaluationProgress: EvaluationProgress | null = null;
  evaluationId: string = '';

  // For displaying result details
  selectedResult: EvaluationResult | null = null;
  showResultDetails = false;

  // Polling for progress updates (if evaluation is running)
  progressPolling = false;

  // Enums for template access
  EvaluationStatus = EvaluationStatus;
  EvaluationMethod = EvaluationMethod;

  // Metrics charting
  metricsData: { name: string, value: number }[] = [];

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
    this.route.paramMap.pipe(
      takeUntil(this.destroy$)
    ).subscribe(params => {
      const id = params.get('id');
      if (id) {
        this.evaluationId = id;
        this.loadEvaluation(id);
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

  loadEvaluation(id: string): void {
    this.isLoading = true;
    this.error = null;

    this.evaluationService.getEvaluation(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (evaluation) => {
          this.evaluation = evaluation;
          this.isLoading = false;

          // Load agent, dataset, and prompt names
          this.loadRelatedEntities();

          if (this.evaluation.results && this.evaluation.results.length > 0) {
            this.prepareMetricsData();
          }

          // Start polling for progress if evaluation is running
          if (evaluation.status === EvaluationStatus.RUNNING) {
            this.startProgressPolling();
          } else if (evaluation.status === EvaluationStatus.PENDING) {
            // For pending evaluations, fetch initial progress once
            this.loadEvaluationProgress();
          }
        },
        error: (error) => {
          this.error = 'Failed to load evaluation details. Please try again.';
          this.isLoading = false;
          console.error('Error loading evaluation:', error);
        }
      });
  }

  loadRelatedEntities(): void {
    if (!this.evaluation) return;

    // Load agent details if we have agent_id
    if (this.evaluation.agent_id && !this.evaluation.agent) {
      this.agentService.getAgent(this.evaluation.agent_id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (agent) => {
            this.evaluation!.agent = agent;
          },
          error: (error) => {
            console.error('Error loading agent details:', error);
          }
        });
    }

    // Load dataset details if we have dataset_id
    if (this.evaluation.dataset_id && !this.evaluation.dataset) {
      this.datasetService.getDataset(this.evaluation.dataset_id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (dataset) => {
            this.evaluation!.dataset = dataset as unknown as Dataset;
          },
          error: (error) => {
            console.error('Error loading dataset details:', error);
          }
        });
    }

    // Load prompt details if we have prompt_id
    if (this.evaluation.prompt_id && !this.evaluation.prompt) {
      this.promptService.getPromptById(this.evaluation.prompt_id)
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
              // Add any other required properties
            } as Prompt;
          },
          error: (error) => {
            console.error('Error loading prompt details:', error);
          }
        });
    }
  }

  editEvaluation(): void {
    if (this.evaluation) {
      this.router.navigate(['app/evaluations', this.evaluation.id, 'edit']);
    }
  }

  startEvaluation(): void {
    if (!this.evaluation) return;

    this.confirmationDialogService.confirm({
      title: 'Start Evaluation',
      message: 'Are you sure you want to start this evaluation?',
      confirmText: 'Start',
      cancelText: 'Cancel',
      type: 'info'
    }).subscribe(confirmed => {
      if (confirmed) {
        this.evaluationService.startEvaluation(this.evaluation!.id)
          .pipe(takeUntil(this.destroy$))
          .subscribe({
            next: (evaluation) => {
              this.notificationService.success('Evaluation started successfully');
              // Refresh evaluation details
              this.loadEvaluation(this.evaluation!.id);
              this.startProgressPolling();
            },
            error: (error) => {
              this.notificationService.error('Failed to start evaluation. Please try again.');
              console.error('Error starting evaluation:', error);
            }
          });
      }
    });
  }

  cancelEvaluation(): void {
    if (!this.evaluation) return;

    this.confirmationDialogService.confirm({
      title: 'Cancel Evaluation',
      message: 'Are you sure you want to cancel this evaluation?',
      confirmText: 'Cancel Evaluation',
      cancelText: 'Keep Running',
      type: 'warning'
    }).subscribe(confirmed => {
      if (confirmed) {
        this.evaluationService.cancelEvaluation(this.evaluation!.id)
          .pipe(takeUntil(this.destroy$))
          .subscribe({
            next: (evaluation) => {
              this.notificationService.success('Evaluation cancelled successfully');
              // Refresh evaluation details
              this.loadEvaluation(this.evaluation!.id);
              this.stopProgressPolling();
            },
            error: (error) => {
              this.notificationService.error('Failed to cancel evaluation. Please try again.');
              console.error('Error cancelling evaluation:', error);
            }
          });
      }
    });
  }

  deleteEvaluation(): void {
    if (!this.evaluation) return;

    this.confirmationDialogService.confirmDelete('Evaluation')
      .subscribe(confirmed => {
        if (confirmed) {
          this.evaluationService.deleteEvaluation(this.evaluation!.id)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.notificationService.success('Evaluation deleted successfully');
                // Navigate back to evaluations list
                this.router.navigate(['app/evaluations']);
              },
              error: (error) => {
                this.notificationService.error('Failed to delete evaluation. Please try again.');
                console.error('Error deleting evaluation:', error);
              }
            });
        }
      });
  }

  viewResultDetails(result: any): void {
    this.selectedResult = result;
    this.showResultDetails = true;
  }

  closeResultDetails(): void {
    this.showResultDetails = false;
    this.selectedResult = null;
  }

  loadEvaluationProgress(): void {
    if (!this.evaluation) return;

    this.evaluationService.getEvaluationProgress(this.evaluation.id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (progress) => {
          this.evaluationProgress = progress;
        },
        error: (error) => {
          console.error('Error loading evaluation progress:', error);
        }
      });
  }

  startProgressPolling(): void {
    this.progressPolling = true;

    interval(5000) // Poll every 5 seconds
      .pipe(
        takeUntil(this.destroy$),
        filter(() => this.progressPolling),
        switchMap(() => this.evaluationService.getEvaluationProgress(this.evaluationId))
      )
      .subscribe({
        next: (progress) => {
          this.evaluationProgress = progress;

          // If evaluation status is no longer running, refresh the evaluation data
          // and stop polling
          if (progress['status'] !== EvaluationStatus.RUNNING) {
            this.loadEvaluation(this.evaluationId);
            this.stopProgressPolling();
          }
        },
        error: (error) => {
          console.error('Error polling evaluation progress:', error);
        }
      });
  }

  stopProgressPolling(): void {
    this.progressPolling = false;
  }

  navigateToCreateReport(): void {
    // Navigate to report creation page with evaluation ID
    // This will be implemented when we add the reports module
    this.notificationService.info('Report creation feature will be implemented in the Reports module');
  }

  prepareMetricsData(): void {
    if (!this.evaluation || !this.evaluation.results || this.evaluation.results.length === 0) {
      return;
    }

    // Collect all metrics across all results
    const metricNames = new Set<string>();
    const metricValues: Record<string, number[]> = {};

    this.evaluation.results.forEach(result => {
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

    // Calculate average for each metric
    this.metricsData = Array.from(metricNames).map(name => {
      const values = metricValues[name];
      const average = values.reduce((sum, value) => sum + value, 0) / values.length;
      return {
        name,
        value: parseFloat(average.toFixed(2))
      };
    });

    // Sort by value descending
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
        minute: '2-digit'
      }).format(date);
    } catch (e) {
      return 'Invalid date';
    }
  }

  formatDuration(startTime: string | undefined, endTime: string | undefined): string {
    if (!startTime || !endTime) return 'N/A';

    try {
      const start = new Date(startTime).getTime();
      const end = new Date(endTime).getTime();
      const durationMs = end - start;

      if (durationMs < 0) return 'Invalid duration';

      // Format duration
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

  /**
   * Check if evaluation can be started (only PENDING evaluations)
   */
  canStartEvaluation(): boolean {
    return this.evaluation?.status === EvaluationStatus.PENDING;
  }

  /**
   * Check if evaluation can be cancelled (only RUNNING evaluations)
   */
  canCancelEvaluation(): boolean {
    return this.evaluation?.status === EvaluationStatus.RUNNING;
  }

  /**
   * Check if evaluation has results to show
   */
  hasResults(): boolean {
    return !!this.evaluation?.results && this.evaluation.results.length > 0;
  }

  /**
   * Navigate to detailed view for related data
   */
  viewAgent(): void {
    if (this.evaluation?.agent_id) {
      this.router.navigate(['app/agents', this.evaluation.agent_id]);
    }
  }

  viewDataset(): void {
    if (this.evaluation?.dataset_id) {
      this.router.navigate(['app/datasets/datasets', this.evaluation.dataset_id]);
    }
  }

  viewPrompt(): void {
    if (this.evaluation?.prompt_id) {
      this.router.navigate(['app/prompts', this.evaluation.prompt_id]);
    }
  }
}