/* Path: libs/feature/llm-eval/src/lib/pages/evaluations/evaluations.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';
import {
  Evaluation,
  EvaluationFilterParams,
  EvaluationStatus,
  EvaluationMethod
} from '@ngtx-apps/data-access/models';
import { EvaluationService } from '@ngtx-apps/data-access/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent
} from '@ngtx-apps/ui/components';
import {
  AlertService,
  ConfirmationDialogService,
  NotificationService
} from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-evaluations',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracButtonComponent,
    QracTextBoxComponent,
    QracSelectComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './evaluations.page.html',
  styleUrls: ['./evaluations.page.scss']
})
export class EvaluationsPage implements OnInit, OnDestroy {
  evaluations: Evaluation[] = [];
  totalCount = 0;
  isLoading = false;
  error: string | null = null;
  currentPage = 1;
  itemsPerPage = 10; // Updated from 5 to match standard
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: EvaluationFilterParams = {
    page: 1,
    limit: 10, // Updated from 5 to match standard
    sortBy: 'created_at',
    sortDirection: 'desc'
  };

  // For filtering by status
  statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: EvaluationStatus.PENDING, label: 'Pending' },
    { value: EvaluationStatus.RUNNING, label: 'Running' },
    { value: EvaluationStatus.COMPLETED, label: 'Completed' },
    { value: EvaluationStatus.FAILED, label: 'Failed' },
    { value: EvaluationStatus.CANCELLED, label: 'Cancelled' }
  ];

  // Method options
  methodOptions = [
    { value: '', label: 'All Methods' },
    { value: EvaluationMethod.RAGAS, label: 'RAGAS' },
    { value: EvaluationMethod.DEEPEVAL, label: 'DeepEval' },
    { value: EvaluationMethod.CUSTOM, label: 'Custom' },
    { value: EvaluationMethod.MANUAL, label: 'Manual' }
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private evaluationService: EvaluationService,
    private alertService: AlertService,
    private confirmationDialogService: ConfirmationDialogService,
    private notificationService: NotificationService,
    private router: Router,
    private fb: FormBuilder
  ) {
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      method: ['']
    });
  }

  ngOnInit(): void {
    this.setupFilterListeners();
    this.loadEvaluations();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  setupFilterListeners(): void {
    // Set up search debounce
    this.filterForm.get('search')?.valueChanges
      .pipe(
        debounceTime(400),
        distinctUntilChanged(),
        takeUntil(this.destroy$)
      )
      .subscribe((value: string) => {
        this.filterParams.name = value || undefined;
        this.filterParams.page = 1;
        this.loadEvaluations();
      });

    // Listen to status changes
    this.filterForm.get('status')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.status = value ? value as EvaluationStatus : undefined;
        this.filterParams.page = 1;
        this.loadEvaluations();
      });

    // Listen to method changes
    this.filterForm.get('method')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.method = value ? value as EvaluationMethod : undefined;
        this.filterParams.page = 1;
        this.loadEvaluations();
      });
  }

  loadEvaluations(): void {
    this.isLoading = true;
    this.error = null;

    this.evaluationService.getEvaluations(this.filterParams)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.evaluations = response.evaluations;
          this.totalCount = response.totalCount;
          this.isLoading = false;
          this.updateVisiblePages();
        },
        error: (error) => {
          this.error = 'Failed to load evaluations. Please try again.';
          this.alertService.showAlert({
            show: true,
            message: this.error,
            title: 'Error'
          });
          this.isLoading = false;
          console.error('Error loading evaluations:', error);
        }
      });
  }

  updateVisiblePages(): void {
    const maxVisiblePages = 5;
    const totalPages = Math.ceil(this.totalCount / this.itemsPerPage);
    const pages: number[] = [];

    if (totalPages <= maxVisiblePages) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      pages.push(1);

      let startPage = Math.max(2, this.filterParams.page! - 1);
      let endPage = Math.min(totalPages - 1, this.filterParams.page! + 1);

      if (this.filterParams.page! <= 3) {
        endPage = Math.min(totalPages - 1, 4);
      } else if (this.filterParams.page! >= totalPages - 2) {
        startPage = Math.max(2, totalPages - 3);
      }

      if (startPage > 2) {
        pages.push(-1);
      }

      for (let i = startPage; i <= endPage; i++) {
        pages.push(i);
      }

      if (endPage < totalPages - 1) {
        pages.push(-2);
      }

      if (totalPages > 1) {
        pages.push(totalPages);
      }
    }

    this.visiblePages = pages;
  }

  onPageChange(page: number, event: Event): void {
    event.preventDefault();
    if (page < 1) return;

    this.filterParams.page = page;
    this.loadEvaluations();
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      method: ''
    });

    this.filterParams.name = undefined;
    this.filterParams.status = undefined;
    this.filterParams.method = undefined;
    this.filterParams.page = 1;

    this.loadEvaluations();
  }

  onSortChange(sortBy: string): void {
    const validSortFields = ["created_at", "updated_at", "name", "status", "method", "start_time", "end_time"];

    if (validSortFields.includes(sortBy)) {
      if (this.filterParams.sortBy === sortBy) {
        this.filterParams.sortDirection =
          this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        this.filterParams.sortBy = sortBy as "created_at" | "name" | "status" | "updated_at";
        this.filterParams.sortDirection = 'desc';
      }

      this.filterParams.page = 1;
      this.loadEvaluations();
    } else {
      console.warn(`Invalid sort field: ${sortBy}. Using default sort.`);
    }
  }

  onEvaluationClick(evaluation: Evaluation): void {
    this.router.navigate(['app/evaluations', evaluation.id]);
  }

  onEditEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/evaluations', evaluationId, 'edit']);
  }

  createNewEvaluation(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/evaluations/create']);
  }

  confirmDeleteEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation();

    this.confirmationDialogService.confirmDelete('Evaluation')
      .subscribe(confirmed => {
        if (confirmed) {
          this.deleteEvaluation(evaluationId);
        }
      });
  }

  private deleteEvaluation(evaluationId: string): void {
    this.evaluationService.deleteEvaluation(evaluationId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.alertService.showAlert({
            show: true,
            message: 'Evaluation deleted successfully',
            title: 'Success'
          });
          this.loadEvaluations();
        },
        error: (error) => {
          this.alertService.showAlert({
            show: true,
            message: 'Failed to delete evaluation. Please try again.',
            title: 'Error'
          });
          console.error('Error deleting evaluation:', error);
        }
      });
  }

  startEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation();

    this.confirmationDialogService.confirm({
      title: 'Start Evaluation',
      message: 'Are you sure you want to start this evaluation?',
      confirmText: 'Start',
      cancelText: 'Cancel',
      type: 'info'
    }).subscribe(confirmed => {
      if (confirmed) {
        this.evaluationService.startEvaluation(evaluationId)
          .pipe(takeUntil(this.destroy$))
          .subscribe({
            next: () => {
              this.alertService.showAlert({
                show: true,
                message: 'Evaluation started successfully',
                title: 'Success'
              });
              this.loadEvaluations();
            },
            error: (error) => {
              this.alertService.showAlert({
                show: true,
                message: 'Failed to start evaluation. Please try again.',
                title: 'Error'
              });
              console.error('Error starting evaluation:', error);
            }
          });
      }
    });
  }

  cancelEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation();

    this.confirmationDialogService.confirm({
      title: 'Cancel Evaluation',
      message: 'Are you sure you want to cancel this evaluation?',
      confirmText: 'Cancel Evaluation',
      cancelText: 'Keep Running',
      type: 'warning'
    }).subscribe(confirmed => {
      if (confirmed) {
        this.evaluationService.cancelEvaluation(evaluationId)
          .pipe(takeUntil(this.destroy$))
          .subscribe({
            next: () => {
              this.alertService.showAlert({
                show: true,
                message: 'Evaluation cancelled successfully',
                title: 'Success'
              });
              this.loadEvaluations();
            },
            error: (error) => {
              this.alertService.showAlert({
                show: true,
                message: 'Failed to cancel evaluation. Please try again.',
                title: 'Error'
              });
              console.error('Error cancelling evaluation:', error);
            }
          });
      }
    });
  }

  formatDate(dateString: string): string {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      }).format(date);
    } catch (e) {
      return 'Invalid date';
    }
  }

  truncateText(text: string | undefined, maxLength = 100): string {
    if (!text) return '';
    return text.length > maxLength
      ? `${text.substring(0, maxLength)}...`
      : text;
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

  canStartEvaluation(status: EvaluationStatus): boolean {
    return status === EvaluationStatus.PENDING;
  }

  canCancelEvaluation(status: EvaluationStatus): boolean {
    return status === EvaluationStatus.RUNNING;
  }
}