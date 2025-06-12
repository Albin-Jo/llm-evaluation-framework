import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  FormBuilder,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
} from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';
import {
  Evaluation,
  EvaluationFilterParams,
  EvaluationStatus,
  EvaluationMethod,
} from '@ngtx-apps/data-access/models';
import { EvaluationService } from '@ngtx-apps/data-access/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent,
} from '@ngtx-apps/ui/components';
import {
  AlertService,
  ConfirmationDialogService,
  NotificationService,
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
    QracSelectComponent,
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './evaluations.page.html',
  styleUrls: ['./evaluations.page.scss'],
})
export class EvaluationsPage implements OnInit, OnDestroy {
  evaluations: Evaluation[] = [];
  totalCount = 0;
  isLoading = false;
  error: string | null = null;
  currentPage = 1;
  itemsPerPage = 5;
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: EvaluationFilterParams = {
    page: 1,
    limit: 5,
    sortBy: 'created_at',
    sortDirection: 'desc',
  };

  // For filtering by status
  statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: EvaluationStatus.PENDING, label: 'Pending' },
    { value: EvaluationStatus.RUNNING, label: 'Running' },
    { value: EvaluationStatus.COMPLETED, label: 'Completed' },
    { value: EvaluationStatus.FAILED, label: 'Failed' },
    { value: EvaluationStatus.CANCELLED, label: 'Cancelled' },
  ];

  // Method options
  methodOptions = [
    { value: '', label: 'All Methods' },
    { value: EvaluationMethod.RAGAS, label: 'RAGAS' },
    { value: EvaluationMethod.DEEPEVAL, label: 'DeepEval' },
    { value: EvaluationMethod.CUSTOM, label: 'Custom' },
    { value: EvaluationMethod.MANUAL, label: 'Manual' },
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
    // Initialize form with proper structure
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      method: [''],
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

  // ISSUE 5 FIX: Enhanced filter setup with proper error handling
  setupFilterListeners(): void {
    // Set up search debounce with error handling
    this.filterForm
      .get('search')
      ?.valueChanges.pipe(
        debounceTime(400),
        distinctUntilChanged(),
        takeUntil(this.destroy$)
      )
      .subscribe({
        next: (value: string) => {
          console.log('Search filter changed:', value); // Debug log
          this.filterParams.name = value?.trim() || undefined;
          this.filterParams.page = 1;
          this.loadEvaluations();
        },
        error: (error) => {
          console.error('Error in search filter:', error);
        },
      });

    // Listen to status changes with validation
    this.filterForm
      .get('status')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (value: string) => {
          console.log('Status filter changed:', value); // Debug log
          this.filterParams.status =
            value && value.trim() !== ''
              ? (value as EvaluationStatus)
              : undefined;
          this.filterParams.page = 1;
          this.loadEvaluations();
        },
        error: (error) => {
          console.error('Error in status filter:', error);
        },
      });

    // Listen to method changes with validation
    this.filterForm
      .get('method')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (value: string) => {
          console.log('Method filter changed:', value); // Debug log
          this.filterParams.method =
            value && value.trim() !== ''
              ? (value as EvaluationMethod)
              : undefined;
          this.filterParams.page = 1;
          this.loadEvaluations();
        },
        error: (error) => {
          console.error('Error in method filter:', error);
        },
      });
  }

  loadEvaluations(): void {
    this.isLoading = true;
    this.error = null;

    // Log current filter params for debugging
    console.log('Loading evaluations with params:', this.filterParams);

    this.evaluationService
      .getEvaluations(this.filterParams)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          console.log('Evaluations loaded:', response); // Debug log
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
            title: 'Error',
          });
          this.isLoading = false;
          console.error('Error loading evaluations:', error);
        },
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

  // Enhanced clear filters with form reset
  clearFilters(): void {
    console.log('Clearing filters'); // Debug log

    // Reset the form controls
    this.filterForm.patchValue(
      {
        search: '',
        status: '',
        method: '',
      },
      { emitEvent: false }
    ); // Don't emit events to avoid double loading

    // Reset filter params
    this.filterParams = {
      page: 1,
      limit: this.itemsPerPage,
      sortBy: 'created_at',
      sortDirection: 'desc',
      // Remove name, status, method to clear filters
    };

    this.loadEvaluations();
  }

  onSortChange(sortBy: string): void {
    const validSortFields = [
      'created_at',
      'updated_at',
      'name',
      'status',
      'method',
      'start_time',
      'end_time',
    ];

    if (validSortFields.includes(sortBy)) {
      if (this.filterParams.sortBy === sortBy) {
        this.filterParams.sortDirection =
          this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        this.filterParams.sortBy = sortBy as
          | 'created_at'
          | 'name'
          | 'status'
          | 'updated_at';
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

  // ISSUE 4 FIX: Add status validation before editing
  onEditEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation();

    // Find the evaluation to check its status
    const evaluation = this.evaluations.find((e) => e.id === evaluationId);

    if (evaluation && evaluation.status !== EvaluationStatus.PENDING) {
      this.notificationService.error(
        `Cannot edit ${evaluation.status.toLowerCase()} evaluation`
      );
      return;
    }

    this.router.navigate(['app/evaluations', evaluationId, 'edit']);
  }

  createNewEvaluation(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/evaluations/create']);
  }

  confirmDeleteEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation();

    this.confirmationDialogService
      .confirmDelete('Evaluation')
      .subscribe((confirmed) => {
        if (confirmed) {
          this.deleteEvaluation(evaluationId);
        }
      });
  }

  private deleteEvaluation(evaluationId: string): void {
    this.evaluationService
      .deleteEvaluation(evaluationId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.notificationService.success('Evaluation deleted successfully');
          this.loadEvaluations();
        },
        error: (error) => {
          this.notificationService.error('Failed to delete evaluation');
          console.error('Error deleting evaluation:', error);
        },
      });
  }

  startEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation();

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
            .startEvaluation(evaluationId)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.notificationService.success(
                  'Evaluation started successfully'
                );
                this.loadEvaluations();
              },
              error: (error) => {
                this.notificationService.error('Failed to start evaluation');
                console.error('Error starting evaluation:', error);
              },
            });
        }
      });
  }

  cancelEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation();

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
            .cancelEvaluation(evaluationId)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.notificationService.success(
                  'Evaluation cancelled successfully'
                );
                this.loadEvaluations();
              },
              error: (error) => {
                this.notificationService.error('Failed to cancel evaluation');
                console.error('Error cancelling evaluation:', error);
              },
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
        day: 'numeric',
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
