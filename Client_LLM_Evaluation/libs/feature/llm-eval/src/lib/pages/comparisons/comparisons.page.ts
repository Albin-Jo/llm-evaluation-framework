import {
  Component,
  OnDestroy,
  OnInit,
  NO_ERRORS_SCHEMA,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  FormBuilder,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
} from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, forkJoin, of, takeUntil } from 'rxjs';
import {
  debounceTime,
  distinctUntilChanged,
  finalize,
  catchError,
} from 'rxjs/operators';
import {
  Comparison,
  ComparisonFilterParams,
  ComparisonStatus,
  Evaluation,
} from '@ngtx-apps/data-access/models';
import {
  ComparisonService,
  EvaluationService,
} from '@ngtx-apps/data-access/services';
import {
  QracTextBoxComponent,
  QracSelectComponent,
  QracButtonComponent,
} from '@ngtx-apps/ui/components';
import {
  ConfirmationDialogService,
  NotificationService,
} from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-comparisons',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracTextBoxComponent,
    QracSelectComponent,
    QracButtonComponent,
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './comparisons.page.html',
  styleUrls: ['./comparisons.page.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ComparisonsPage implements OnInit, OnDestroy {
  comparisons: Comparison[] = [];
  evaluations: Evaluation[] = [];
  totalCount = 0;
  isLoading = false;
  error: string | null = null;
  currentPage = 1;
  itemsPerPage = 10;
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: ComparisonFilterParams = {
    page: 1,
    limit: 10,
    sortBy: 'created_at',
    sortDirection: 'desc',
  };

  // For filtering by status
  statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: ComparisonStatus.PENDING, label: 'Pending' },
    { value: ComparisonStatus.RUNNING, label: 'Running' },
    { value: ComparisonStatus.COMPLETED, label: 'Completed' },
    { value: ComparisonStatus.FAILED, label: 'Failed' },
  ];

  // For filtering by evaluation
  evaluationOptions = [{ value: '', label: 'All Evaluations' }];

  // Cache for evaluation names
  evaluationNameCache: Record<string, string> = {};

  private destroy$ = new Subject<void>();

  constructor(
    private comparisonService: ComparisonService,
    private evaluationService: EvaluationService,
    private confirmationDialogService: ConfirmationDialogService,
    private notificationService: NotificationService,
    private router: Router,
    private fb: FormBuilder,
    private cdr: ChangeDetectorRef
  ) {
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      evaluation: [''],
    });
  }

  ngOnInit(): void {
    this.loadEvaluationsForFilter();
    this.setupFilterListeners();
    this.loadComparisons();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Load evaluations for the filter dropdown
   */
  loadEvaluationsForFilter(): void {
    this.evaluationService
      .getEvaluations({ limit: 100 })
      .pipe(
        takeUntil(this.destroy$),
        catchError((error) => {
          console.error('Error loading evaluations for filter:', error);
          return of({ evaluations: [], totalCount: 0 });
        })
      )
      .subscribe((response) => {
        this.evaluations = response.evaluations;

        // Create a mapping of evaluation IDs to names for quick lookup
        this.evaluations.forEach((evaluation) => {
          this.evaluationNameCache[evaluation.id] = evaluation.name;
        });

        // Populate filter options
        this.evaluationOptions = [
          { value: '', label: 'All Evaluations' },
          ...this.evaluations.map((evaluation) => ({
            value: evaluation.id,
            label: evaluation.name,
          })),
        ];

        this.cdr.markForCheck();
      });
  }

  setupFilterListeners(): void {
    // Set up search debounce
    this.filterForm
      .get('search')
      ?.valueChanges.pipe(
        debounceTime(400),
        distinctUntilChanged(),
        takeUntil(this.destroy$)
      )
      .subscribe((value: string) => {
        this.filterParams.name = value || undefined;
        this.filterParams.page = 1;
        this.loadComparisons();
      });

    // Listen to status changes
    this.filterForm
      .get('status')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.status = value
          ? (value as ComparisonStatus)
          : undefined;
        this.filterParams.page = 1;
        this.loadComparisons();
      });

    // Listen to evaluation changes
    this.filterForm
      .get('evaluation')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        if (value) {
          // When an evaluation is selected, we want to find comparisons that have it
          // in either evaluation_a_id or evaluation_b_id, but the backend may not support this directly.
          // We'll handle this in the implementation when it's clear how the backend API works.
          // For now, we'll just store the evaluation ID in a custom field.
          this.filterParams.evaluation_a_id = value;
          this.filterParams.evaluation_b_id = value;
        } else {
          this.filterParams.evaluation_a_id = undefined;
          this.filterParams.evaluation_b_id = undefined;
        }
        this.filterParams.page = 1;
        this.loadComparisons();
      });
  }

  loadComparisons(): void {
    this.isLoading = true;
    this.error = null;
    this.cdr.markForCheck();

    this.comparisonService
      .getComparisons(this.filterParams)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isLoading = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe({
        next: (response) => {
          this.comparisons = response.comparisons;
          this.totalCount = response.totalCount;
          this.updateVisiblePages();
          this.cdr.markForCheck();
        },
        error: (error) => {
          this.error = 'Failed to load comparisons. Please try again.';
          this.notificationService.error(this.error);
          console.error('Error loading comparisons:', error);
          this.cdr.markForCheck();
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
    this.loadComparisons();
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      evaluation: '',
    });

    this.filterParams.name = undefined;
    this.filterParams.status = undefined;
    this.filterParams.evaluation_a_id = undefined;
    this.filterParams.evaluation_b_id = undefined;
    this.filterParams.page = 1;

    this.loadComparisons();
  }

  onSortChange(sortBy: string): void {
    const validSortFields = ['created_at', 'updated_at', 'name', 'status'];

    if (validSortFields.includes(sortBy)) {
      if (this.filterParams.sortBy === sortBy) {
        this.filterParams.sortDirection =
          this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        this.filterParams.sortBy = sortBy;
        this.filterParams.sortDirection = 'desc';
      }

      this.filterParams.page = 1;
      this.loadComparisons();
    }
  }

  onComparisonClick(comparison: Comparison): void {
    this.router.navigate(['app/comparisons', comparison.id]);
  }

  onViewComparison(event: Event, comparisonId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/comparisons', comparisonId]);
  }

  onEditComparison(event: Event, comparisonId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/comparisons', comparisonId, 'edit']);
  }

  createNewComparison(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/comparisons/create']);
  }

  confirmDeleteComparison(event: Event, comparisonId: string): void {
    event.stopPropagation();

    this.confirmationDialogService
      .confirmDelete('Comparison')
      .subscribe((confirmed) => {
        if (confirmed) {
          this.deleteComparison(comparisonId);
        }
      });
  }

  private deleteComparison(comparisonId: string): void {
    this.comparisonService
      .deleteComparison(comparisonId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.notificationService.success('Comparison deleted successfully');
          this.loadComparisons();
        },
        error: (error) => {
          this.notificationService.error(
            'Failed to delete comparison. Please try again.'
          );
          console.error('Error deleting comparison:', error);
        },
      });
  }

  runComparison(event: Event, comparisonId: string): void {
    event.stopPropagation();

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
          this.comparisonService
            .runComparison(comparisonId)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.notificationService.success(
                  'Comparison started successfully'
                );
                this.loadComparisons();
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

  formatDate(dateString: string | undefined): string {
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

  canRunComparison(status: ComparisonStatus): boolean {
    return (
      status === ComparisonStatus.PENDING || status === ComparisonStatus.FAILED
    );
  }

  hasResults(comparison: Comparison): boolean {
    return (
      comparison.status === ComparisonStatus.COMPLETED && !!comparison.summary
    );
  }

  getFormattedResult(comparison: Comparison): string {
    if (
      !comparison.summary ||
      comparison.status !== ComparisonStatus.COMPLETED
    ) {
      return 'N/A';
    }

    // Get the overall improvement from the summary
    const improvement = comparison.summary['overall_improvement'] || 0;

    // Format as percentage with sign
    const sign = improvement > 0 ? '+' : '';
    return `${sign}${(improvement * 100).toFixed(1)}%`;
  }

  getResultClass(comparison: Comparison): string {
    if (
      !comparison.summary ||
      comparison.status !== ComparisonStatus.COMPLETED
    ) {
      return 'neutral';
    }

    const improvement = comparison.summary['overall_improvement'] || 0;

    if (improvement > 0) {
      return 'improved';
    } else if (improvement < 0) {
      return 'regressed';
    } else {
      return 'neutral';
    }
  }

  getEvaluationName(evaluationId: string): string {
    // Use the cache if available
    if (this.evaluationNameCache[evaluationId]) {
      return this.evaluationNameCache[evaluationId];
    }

    // Otherwise return a placeholder
    return evaluationId.substring(0, 8) + '...';
  }

  trackByComparisonId(index: number, comparison: Comparison): string {
    return comparison.id;
  }
}
