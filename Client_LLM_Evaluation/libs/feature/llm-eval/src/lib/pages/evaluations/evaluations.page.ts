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
  QracTagButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent
} from '@ngtx-apps/ui/components';
import {
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
  itemsPerPage = 5;
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: EvaluationFilterParams = {
    page: 1,
    limit: 5,
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
        // Update filter params for search
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

    console.log('Loading evaluations with params:', this.filterParams);

    this.evaluationService.getEvaluations(this.filterParams)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.evaluations = response.evaluations;
          this.totalCount = response.totalCount;
          this.isLoading = false;

          console.log(`Loaded ${this.evaluations.length} evaluations. Total: ${this.totalCount}`);

          // Calculate pagination
          this.updateVisiblePages();
        },
        error: (error) => {
          this.error = 'Failed to load evaluations. Please try again.';
          this.notificationService.error(this.error);
          this.isLoading = false;
          console.error('Error loading evaluations:', error);
        }
      });
  }

  /**
   * Update the array of visible page numbers
   */
  updateVisiblePages(): void {
    const maxVisiblePages = 5;
    const totalPages = Math.ceil(this.totalCount / this.itemsPerPage);
    const pages: number[] = [];

    console.log(`Updating pagination. Total pages: ${totalPages}, Current page: ${this.filterParams.page}`);

    if (totalPages <= maxVisiblePages) {
      // If total pages are less than max visible, show all pages
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Always show first page
      pages.push(1);

      let startPage = Math.max(2, this.filterParams.page! - 1);
      let endPage = Math.min(totalPages - 1, this.filterParams.page! + 1);

      // Adjust if we're near the start or end
      if (this.filterParams.page! <= 3) {
        endPage = Math.min(totalPages - 1, 4);
      } else if (this.filterParams.page! >= totalPages - 2) {
        startPage = Math.max(2, totalPages - 3);
      }

      // Add ellipsis if needed
      if (startPage > 2) {
        pages.push(-1); // -1 represents ellipsis
      }

      // Add middle pages
      for (let i = startPage; i <= endPage; i++) {
        pages.push(i);
      }

      // Add ellipsis if needed
      if (endPage < totalPages - 1) {
        pages.push(-2); // -2 represents ellipsis
      }

      // Always show last page
      if (totalPages > 1) {
        pages.push(totalPages);
      }
    }

    this.visiblePages = pages;
    console.log('Visible pages:', this.visiblePages);
  }

  onPageChange(page: number, event: Event): void {
    event.preventDefault();
    if (page < 1) return;

    console.log(`Changing to page ${page}`);
    this.filterParams.page = page;
    this.loadEvaluations();
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      method: ''
    });

    // Reset filter params manually
    this.filterParams.name = undefined;
    this.filterParams.status = undefined;
    this.filterParams.method = undefined;
    this.filterParams.page = 1;

    console.log('Filters cleared');
    this.loadEvaluations();
  }

  onSortChange(sortBy: string): void {
    console.log(`Sorting by ${sortBy}, current sort: ${this.filterParams.sortBy}, direction: ${this.filterParams.sortDirection}`);

    // Define all valid sort fields
    const validSortFields = ["created_at", "updated_at", "name", "status", "method", "start_time", "end_time"];

    // Ensure the sort field is valid
    if (validSortFields.includes(sortBy)) {
      if (this.filterParams.sortBy === sortBy) {
        // Toggle direction if same sort field
        this.filterParams.sortDirection =
          this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
        console.log(`Changed sort direction to ${this.filterParams.sortDirection}`);
      } else {
        // Default to desc for new sort field
        this.filterParams.sortBy = sortBy as "created_at" | "name" | "status" | "updated_at";
        this.filterParams.sortDirection = 'desc';
        console.log(`Changed sort field to ${sortBy} with direction desc`);
      }

      // Reset to page 1 when sorting changes
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
    event.stopPropagation(); // Prevent row click
    this.router.navigate(['app/evaluations', evaluationId, 'edit']);
  }

  createNewEvaluation(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/evaluations/create']);
  }

  confirmDeleteEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation(); // Prevent navigation to detail page

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
          this.notificationService.success('Evaluation deleted successfully');
          this.loadEvaluations(); // Reload the list
        },
        error: (error) => {
          this.notificationService.error('Failed to delete evaluation. Please try again.');
          console.error('Error deleting evaluation:', error);
        }
      });
  }

  startEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation(); // Prevent navigation to detail page

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
              this.notificationService.success('Evaluation started successfully');
              this.loadEvaluations(); // Reload the list
            },
            error: (error) => {
              this.notificationService.error('Failed to start evaluation. Please try again.');
              console.error('Error starting evaluation:', error);
            }
          });
      }
    });
  }

  cancelEvaluation(event: Event, evaluationId: string): void {
    event.stopPropagation(); // Prevent navigation to detail page

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
              this.notificationService.success('Evaluation cancelled successfully');
              this.loadEvaluations(); // Reload the list
            },
            error: (error) => {
              this.notificationService.error('Failed to cancel evaluation. Please try again.');
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

  /**
   * Truncate text to specified length
   */
  truncateText(text: string | undefined, maxLength = 100): string {
    if (!text) return '';
    return text.length > maxLength
      ? `${text.substring(0, maxLength)}...`
      : text;
  }

  /**
   * Get status badge class based on evaluation status
   */
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

  /**
   * Check if evaluation can be started (only PENDING evaluations)
   */
  canStartEvaluation(status: EvaluationStatus): boolean {
    return status === EvaluationStatus.PENDING;
  }

  /**
   * Check if evaluation can be cancelled (only RUNNING evaluations)
   */
  canCancelEvaluation(status: EvaluationStatus): boolean {
    return status === EvaluationStatus.RUNNING;
  }
}
