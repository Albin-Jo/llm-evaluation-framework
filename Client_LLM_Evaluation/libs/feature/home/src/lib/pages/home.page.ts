/* Path: libs/feature/home/src/lib/pages/home.page.ts */
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { CUSTOM_ELEMENTS_SCHEMA, NO_ERRORS_SCHEMA } from '@angular/core';
import { Component, ElementRef, inject, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { FilterService, FreezeService, GridModule, SortService } from '@syncfusion/ej2-angular-grids';
import { Observable, Subject, forkJoin, of } from 'rxjs';
import { catchError, finalize, takeUntil } from 'rxjs/operators';

import {
  QracButtonComponent,
  QracuploadComponent
} from '@ngtx-apps/ui/components';
import {
  AgentService,
  DatasetService,
  EvaluationService,
  PromptService
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
  PromptsResponse
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
    evaluations: false
  };

  // Feature stats
  datasetsCount = 0;
  promptsCount = 0;
  agentsCount = 0;
  evaluationsCount = 0;

  // Dashboard data
  datasets: Dataset[] = [];
  recentPrompts: PromptResponse[] = [];
  recentAgents: Agent[] = [];
  recentEvaluations: (Evaluation | EvaluationDetail)[] = [];

  // Status enum for template access
  datasetStatus = DatasetStatus;
  evaluationStatus = EvaluationStatus;
  agentStatus = AgentStatus;
  agentDomain = AgentDomain;

  // Service injections
  private readonly idleTimeoutService = inject(IdleTimeoutService);
  private readonly datasetService = inject(DatasetService);
  private readonly promptService = inject(PromptService);
  private readonly agentService = inject(AgentService);
  private readonly evaluationService = inject(EvaluationService);
  private readonly router = inject(Router);

  constructor() {}

  ngOnInit(): void {
    this.idleTimeoutService.subscribeIdletimeout();
    this.loadDashboardData();
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

    // Load datasets
    this.loadDatasetsData();

    // Load prompts
    this.loadPromptsData();

    // Load agents
    this.loadAgentsData();

    // Load evaluations
    this.loadEvaluationsData();
  }

  /**
   * Load datasets data
   */
  private loadDatasetsData(): void {
    this.loadingStates.datasets = true;

    this.datasetService.getDatasets({
      page: 1,
      limit: 100,
      is_public: true
    }).pipe(
      takeUntil(this.destroy$),
      finalize(() => this.loadingStates.datasets = false)
    ).subscribe({
      next: (response) => {
        this.datasetsCount = response.totalCount;
        this.datasets = response.datasets.slice(0, 2); // Get just the first 2 datasets
      },
      error: (err) => {
        console.error('Error loading datasets:', err);
        this.showToast('Failed to load datasets', 'error');
      }
    });
  }

  /**
   * Load prompts data
   */
  private loadPromptsData(): void {
    this.loadingStates.prompts = true;

    this.promptService.getPrompts({
      isPublic: true
    }).pipe(
      takeUntil(this.destroy$),
      finalize(() => this.loadingStates.prompts = false)
    ).subscribe({
      next: (response: PromptsResponse) => {
        this.promptsCount = response.totalCount;
        this.recentPrompts = response.prompts.slice(0, 2); // Just take the 2 most recent prompts
      },
      error: (err) => {
        console.error('Error loading prompts:', err);
        this.showToast('Failed to load prompts', 'error');
      }
    });
  }

  /**
   * Load agents data
   */
  private loadAgentsData(): void {
    this.loadingStates.agents = true;

    this.agentService.getAgents({
      page: 1,
      limit: 100
    }).pipe(
      takeUntil(this.destroy$),
      finalize(() => this.loadingStates.agents = false)
    ).subscribe({
      next: (response) => {
        this.agentsCount = response.totalCount;
        this.recentAgents = response.agents.slice(0, 2); // Get just the first 2 agents
      },
      error: (err) => {
        console.error('Error loading agents:', err);
        this.showToast('Failed to load agents', 'error');
      }
    });
  }

  /**
   * Load evaluations data
   */
  private loadEvaluationsData(): void {
    this.loadingStates.evaluations = true;

    this.evaluationService.getEvaluations({
      page: 1,
      limit: 100
    }).pipe(
      takeUntil(this.destroy$),
      finalize(() => {
        this.loadingStates.evaluations = false;
        this.checkAllLoaded();
      }),
      catchError(err => {
        console.error('Error loading evaluations:', err);
        this.showToast('Failed to load evaluations', 'error');
        return of({ evaluations: [], totalCount: 0 });
      })
    ).subscribe(response => {
      this.evaluationsCount = response.totalCount;

      // When we receive evaluation data, enrich it with additional details
      if (response.evaluations.length > 0) {
        // Get first 2 evaluations and then load details for them
        const evaluationsToShow = response.evaluations.slice(0, 2);

        // Create a mock response for now since we lack actual evaluation details
        // This should be replaced with actual API calls when the endpoint is ready
        this.recentEvaluations = evaluationsToShow.map(evaluation => {
          // Create a mock evaluation detail that extends the base evaluation
          const detailEval = {
            ...evaluation,
            // Add mock metrics_results for completed evaluations
            metrics_results: evaluation.status === EvaluationStatus.COMPLETED ? {
              relevance: Math.random() * 0.5 + 0.5, // Random value between 0.5 and 1.0
              latency: Math.floor(Math.random() * 500) + 200 // Random latency between 200-700ms
            } : undefined,
            // Add mock progress for running evaluations
            progress: evaluation.status === EvaluationStatus.RUNNING ? {
              total: 100,
              completed: Math.floor(Math.random() * 80) + 10, // Random progress between 10-90%
              failed: 0,
              percentage: 0,
              percentage_complete: Math.floor(Math.random() * 80) + 10, // Same as completed
              processed_items: Math.floor(Math.random() * 80) + 10,
              total_items: 100,
              eta_seconds: 120
            } : undefined
          };

          return detailEval;
        });
      } else {
        this.recentEvaluations = [];
      }
    });
  }

  /**
   * Check if all data loading is complete
   */
  private checkAllLoaded(): void {
    if (!this.loadingStates.datasets && !this.loadingStates.prompts &&
        !this.loadingStates.agents && !this.loadingStates.evaluations) {
      this.isLoading = false;
    }
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

  // Report actions (placeholder for future implementation)
  viewAllReports(): void {
    this.showToast('Reports module coming soon!', 'info');
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
    return !!(evaluation as ExtendedEvaluation).metrics_results;
  }

  hasProgress(evaluation: Evaluation | EvaluationDetail): boolean {
    return !!(evaluation as ExtendedEvaluation).progress;
  }

  getMetricValue(evaluation: Evaluation | EvaluationDetail, metric: string): number {
    return (evaluation as ExtendedEvaluation).metrics_results?.[metric] || 0;
  }

  getProgressValue(evaluation: Evaluation | EvaluationDetail): number {
    return (evaluation as ExtendedEvaluation).progress?.percentage_complete || 0;
  }
}

/**
 * Extended evaluation interface with metrics and progress
 */
 interface ExtendedEvaluation extends Evaluation {
  metrics_results?: {
    relevance: number;
    latency: number;
    [key: string]: any;
  };
  progress?: {
    percentage_complete: number;
    [key: string]: any;
  };
}
