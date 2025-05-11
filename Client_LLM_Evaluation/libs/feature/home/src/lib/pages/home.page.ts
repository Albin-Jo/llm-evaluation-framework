/* Path: libs/feature/home/src/lib/pages/home.page.ts */
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { CUSTOM_ELEMENTS_SCHEMA, NO_ERRORS_SCHEMA } from '@angular/core';
import { Component, ElementRef, inject, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { FilterService, FreezeService, GridModule, SortService } from '@syncfusion/ej2-angular-grids';
import { Observable, Subject, forkJoin, of } from 'rxjs';
import { catchError, finalize, retry, takeUntil } from 'rxjs/operators';

import {
  QracButtonComponent,
  QracuploadComponent
} from '@ngtx-apps/ui/components';
import {
  AgentService,
  DatasetService,
  EvaluationService,
  PromptService,
  ReportService
} from '@ngtx-apps/data-access/services';
import { IdleTimeoutService } from '@ngtx-apps/utils/services';
import {
  Agent,
  AgentDomain,
  AgentFilterParams,
  AgentStatus,
  Dataset,
  DatasetStatus,
  Evaluation,
  EvaluationDetail,
  EvaluationMethod,
  EvaluationStatus,
  PromptResponse,
  PromptsResponse,
  Report,
  ReportStatus
} from '@ngtx-apps/data-access/models';

/**
 * Dashboard component for the home page
 * Displays summary statistics and recent items for key entities
 */
@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
  imports: [
    CommonModule,
    NgFor,
    FormsModule,
    GridModule,
    NgIf,
    QracButtonComponent,
    ReactiveFormsModule,
    RouterModule
  ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA, NO_ERRORS_SCHEMA],
  providers: [SortService, FilterService, FreezeService],
  standalone: true
})
export class HomePage implements OnInit, OnDestroy {
  // Component lifecycle management
  private destroy$ = new Subject<void>();

  // Modal and UI references
  @ViewChild('qrscsystoast', { read: ElementRef, static: false }) qrscsystoast?: ElementRef;

  // UI state
  isLoading = false;
  loadingStates = {
    datasets: false,
    prompts: false,
    agents: false,
    evaluations: false,
    reports: false
  };

  // Feature stats
  datasetsCount = 0;
  promptsCount = 0;
  agentsCount = 0;
  evaluationsCount = 0;
  reportsCount = 0;

  // Dashboard data
  datasets: Dataset[] = [];
  recentPrompts: PromptResponse[] = [];
  recentAgents: Agent[] = [];
  recentEvaluations: (Evaluation | EvaluationDetail)[] = [];
  recentReports: Report[] = [];

  // Status enum for template access
  datasetStatus = DatasetStatus;
  evaluationStatus = EvaluationStatus;
  agentStatus = AgentStatus;
  agentDomain = AgentDomain;
  reportStatus = ReportStatus;

  // Service injections
  private readonly idleTimeoutService = inject(IdleTimeoutService);
  private readonly datasetService = inject(DatasetService);
  private readonly promptService = inject(PromptService);
  private readonly agentService = inject(AgentService);
  private readonly evaluationService = inject(EvaluationService);
  private readonly reportService = inject(ReportService);
  private readonly router = inject(Router);

  constructor() {}

  ngOnInit(): void {
    this.idleTimeoutService.subscribeIdletimeout();
    
    // Add a small delay to ensure component is fully initialized
    setTimeout(() => {
      this.loadDashboardData();
    }, 0);
  }

  ngOnDestroy(): void {
    // Clean up subscriptions
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Loads all dashboard data from various services
   */
  loadDashboardData(): void {
    this.isLoading = true;

    // Use forkJoin to load all data in parallel with better error handling
    forkJoin({
      datasets: this.datasetService.getDatasets({
        page: 1,
        limit: 100,
        is_public: true
      }).pipe(
        retry(2), // Retry failed requests up to 2 times
        catchError(err => {
          console.error('Error loading datasets:', err);
          this.showToast('Failed to load datasets', 'error');
          return of({ datasets: [], totalCount: 0 });
        })
      ),
      
      prompts: this.promptService.getPrompts({
        isPublic: true
      }).pipe(
        retry(2),
        catchError(err => {
          console.error('Error loading prompts:', err);
          this.showToast('Failed to load prompts', 'error');
          return of({ prompts: [], totalCount: 0 });
        })
      ),
      
      agents: this.agentService.getAgents({
        page: 1,
        limit: 100
      }).pipe(
        retry(2),
        catchError(err => {
          console.error('Error loading agents:', err);
          this.showToast('Failed to load agents', 'error');
          return of({ agents: [], totalCount: 0 });
        })
      ),
      
      evaluations: this.evaluationService.getEvaluations({
        page: 1,
        limit: 100
      }).pipe(
        retry(2),
        catchError(err => {
          console.error('Error loading evaluations:', err);
          this.showToast('Failed to load evaluations', 'error');
          return of({ evaluations: [], totalCount: 0 });
        })
      ),
      
      reports: this.reportService.getReports({
        page: 1,
        limit: 100
      }).pipe(
        retry(2),
        catchError(err => {
          console.error('Error loading reports:', err);
          this.showToast('Failed to load reports', 'error');
          return of({ reports: [], totalCount: 0 });
        })
      )
    }).pipe(
      takeUntil(this.destroy$),
      finalize(() => {
        this.isLoading = false;
        // Reset all loading states
        Object.keys(this.loadingStates).forEach(key => {
          this.loadingStates[key as keyof typeof this.loadingStates] = false;
        });
      })
    ).subscribe({
      next: (results) => {
        // Process datasets
        this.datasetsCount = results.datasets.totalCount || 0;
        this.datasets = results.datasets.datasets.slice(0, 2);

        // Process prompts
        this.promptsCount = results.prompts.totalCount || 0;
        this.recentPrompts = results.prompts.prompts.slice(0, 2);

        // Process agents
        this.agentsCount = results.agents.totalCount || 0;
        this.recentAgents = results.agents.agents.slice(0, 2);

        // Process evaluations
        this.evaluationsCount = results.evaluations.totalCount || 0;
        
        // Get the first 2 evaluations and their details if needed
        const evaluationsToDisplay = results.evaluations.evaluations.slice(0, 2);
        
        // For each evaluation, get detailed data if needed
        const detailRequests = evaluationsToDisplay.map(evaluation => {
          // If the evaluation is running or completed, try to get progress or results
          if (evaluation.status === EvaluationStatus.RUNNING) {
            return this.evaluationService.getEvaluationProgress(evaluation.id).pipe(
              catchError(() => of(null))
            );
          } else if (evaluation.status === EvaluationStatus.COMPLETED) {
            return this.evaluationService.getEvaluation(evaluation.id).pipe(
              catchError(() => of(evaluation))
            );
          }
          return of(evaluation);
        });

        // Load additional evaluation details
        if (detailRequests.length > 0) {
          forkJoin(detailRequests).subscribe(detailedEvaluations => {
            this.recentEvaluations = detailedEvaluations.map((detail, index) => {
              const baseEvaluation = evaluationsToDisplay[index];
              
              // If we got progress data for a running evaluation
              if (detail && 'percentage_complete' in detail) {
                return {
                  ...baseEvaluation,
                  progress: detail
                };
              }
              
              // If we got full evaluation details
              if (detail && 'results' in detail) {
                return detail as EvaluationDetail;
              }
              
              // Return the base evaluation
              return baseEvaluation;
            });
          });
        } else {
          this.recentEvaluations = [];
        }

        // Process reports
        this.reportsCount = results.reports.totalCount || 0;
        this.recentReports = results.reports.reports.slice(0, 2);
      },
      error: (err) => {
        console.error('Dashboard loading failed:', err);
        this.showToast('Failed to load dashboard data', 'error');
      }
    });
  }

  // Dataset actions
  createDataset(): void {
    this.router.navigate(['app/datasets/datasets/upload']);
  }

  viewDataset(dataset: Dataset): void {
    this.router.navigate(['app/datasets/datasets', dataset.id]);
  }

  viewAllDatasets(): void {
    this.router.navigate(['app/datasets']);
  }

  // Prompt actions
  createPrompt(): void {
    this.router.navigate(['app/prompts/create']);
  }

  viewPrompt(prompt: PromptResponse): void {
    this.router.navigate(['app/prompts', prompt.id]);
  }

  viewAllPrompts(): void {
    this.router.navigate(['app/prompts']);
  }

  // Agent actions
  createAgent(): void {
    this.router.navigate(['app/agents/create']);
  }

  viewAgent(agent: Agent): void {
    this.router.navigate(['app/agents', agent.id]);
  }

  viewAllAgents(): void {
    this.router.navigate(['app/agents']);
  }

  // Evaluation actions
  startNewEvaluation(): void {
    this.router.navigate(['app/evaluations/create']);
  }

  viewEvaluation(evaluation: Evaluation | EvaluationDetail): void {
    this.router.navigate(['app/evaluations', evaluation.id]);
  }

  viewAllEvaluations(): void {
    this.router.navigate(['app/evaluations']);
  }

  // Report actions
  createReport(): void {
    this.router.navigate(['app/reports/create']);
  }

  viewReport(report: Report): void {
    this.router.navigate(['app/reports', report.id]);
  }

  viewAllReports(): void {
    this.router.navigate(['app/reports']);
  }

  /**
   * Helper to format file size with appropriate units
   */
  formatFileSize(bytes?: number): string {
    if (!bytes) return '0 B';

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  /**
   * Toast notification helper
   */
  showToast(message: string, type: 'success' | 'error' | 'warning' | 'info' = 'success', title = ''): void {
    let toastTitle = title;
    if (!title) {
      switch (type) {
        case 'success': toastTitle = 'Success!'; break;
        case 'error': toastTitle = 'Error!'; break;
        case 'warning': toastTitle = 'Warning!'; break;
        case 'info': toastTitle = 'Information'; break;
      }
    }

    if (this.qrscsystoast && this.qrscsystoast.nativeElement) {
      this.qrscsystoast.nativeElement.presentToast(
        message,
        type,
        toastTitle,
        5000
      );
    } else {
      console.log(`${toastTitle}: ${message}`);
    }
  }

  // Type guard functions for template
  hasMetricsResults(evaluation: Evaluation | EvaluationDetail): boolean {
    return 'results' in evaluation && Array.isArray(evaluation.results) && evaluation.results.length > 0;
  }

  hasProgress(evaluation: Evaluation | EvaluationDetail): boolean {
    return 'progress' in evaluation && evaluation.progress !== undefined;
  }

  getMetricValue(evaluation: Evaluation | EvaluationDetail, metric: string): number {
    if (this.hasMetricsResults(evaluation)) {
      const results = (evaluation as EvaluationDetail).results;
      if (results && results.length > 0) {
        // Find the specific metric value from the results
        const result = results[0];
        if (result['metric_scores']) {
          const metricScore = result['metric_scores'].find((m: any) => m.name === metric);
          return metricScore ? metricScore.value : 0;
        }
      }
    }
    return 0;
  }

  getProgressValue(evaluation: Evaluation | EvaluationDetail): number {
    if (this.hasProgress(evaluation)) {
      return (evaluation as any).progress.percentage_complete || 0;
    }
    return 0;
  }
}

/**
 * Extended evaluation interface with metrics and progress
 */
interface ExtendedEvaluation extends Evaluation {
  progress?: {
    percentage_complete: number;
    [key: string]: any;
  };
}