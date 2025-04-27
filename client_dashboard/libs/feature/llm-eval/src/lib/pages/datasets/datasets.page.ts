/* Path: libs/feature/llm-eval/src/lib/pages/datasets/datasets.page.ts */
import { Component, OnDestroy, OnInit, ViewChild, ElementRef, NgZone, ChangeDetectionStrategy, ChangeDetectorRef, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, Observable, BehaviorSubject, of, combineLatest, EMPTY } from 'rxjs';
import { takeUntil, debounceTime, distinctUntilChanged, catchError, map, tap, switchMap, finalize, startWith, shareReplay } from 'rxjs/operators';
import { ScrollingModule, CdkVirtualScrollViewport } from '@angular/cdk/scrolling';

import {
  Dataset,
  DatasetFilterParams,
  DatasetStatus
} from '@ngtx-apps/data-access/models';
import { DatasetService } from '@ngtx-apps/data-access/services';
import {
  QracButtonComponent,
  QracTagButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent
} from '@ngtx-apps/ui/components';
import { AlertService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-datasets',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    ScrollingModule,
    QracButtonComponent,
    QracTextBoxComponent,
    QracSelectComponent,
    QracTagButtonComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './datasets.page.html',
  styleUrls: ['./datasets.page.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class DatasetsPage implements OnInit, OnDestroy {
  @ViewChild(CdkVirtualScrollViewport) viewport!: CdkVirtualScrollViewport;

  // Loading and error states
  isLoading$ = new BehaviorSubject<boolean>(false);
  error$ = new BehaviorSubject<string | null>(null);
  totalCount$ = new BehaviorSubject<number>(0);

  // Dataset collection
  datasets$ = new BehaviorSubject<Dataset[]>([]);

  // Filter state
  filterForm: FormGroup;
  filterParams$ = new BehaviorSubject<DatasetFilterParams>({
    page: 1,
    limit: 20,
    sortBy: 'createdAt',
    sortDirection: 'desc',
    is_public: true
  });

  // Constants
  Math = Math;
  itemsPerPage = 20;

  // Options for select dropdowns
  statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: DatasetStatus.READY, label: 'Ready' },
    { value: DatasetStatus.PROCESSING, label: 'Processing' },
    { value: DatasetStatus.ERROR, label: 'Error' }
  ];

  // Format options
  formatOptions = [
    { value: '', label: 'All Types' },
    { value: 'csv', label: 'CSV' },
    { value: 'jsonl', label: 'JSONL' },
    { value: 'txt', label: 'Text' },
    { value: 'custom', label: 'Custom' }
  ];

  // Visibility options
  visibilityOptions = [
    { value: 'true', label: 'Public' },
    { value: 'false', label: 'Private' },
    { value: '', label: 'All' }
  ];

  // Date range options
  dateRangeOptions = [
    { value: '', label: 'Any Time' },
    { value: 'today', label: 'Today' },
    { value: 'yesterday', label: 'Yesterday' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' },
    { value: 'custom', label: 'Custom Range' }
  ];

  // Size range options
  sizeRangeOptions = [
    { value: '', label: 'Any Size' },
    { value: 'small', label: 'Small (<1MB)' },
    { value: 'medium', label: 'Medium (1-10MB)' },
    { value: 'large', label: 'Large (>10MB)' }
  ];

  // Pagination related
  visiblePages: number[] = [];

  // Cache to optimize performance
  private cache = new Map<string, { data: Dataset[], totalCount: number, timestamp: number }>();
  private cacheExpiryMs = 5 * 60 * 1000; // 5 minutes

  private destroy$ = new Subject<void>();

  constructor(
    private datasetService: DatasetService,
    private alertService: AlertService,
    private router: Router,
    private fb: FormBuilder,
    private cdr: ChangeDetectorRef,
    private ngZone: NgZone
  ) {
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      type: [''],
      isPublic: ['true'],
      dateRange: [''],
      sizeRange: ['']
    });
  }

  ngOnInit(): void {
    this.setupFilterListeners();

    // Subscribe to filter changes to load datasets
    this.filterParams$
      .pipe(
        takeUntil(this.destroy$),
        switchMap(params => this.loadDatasetsFromCache(params))
      )
      .subscribe({
        next: ({ data, totalCount }) => {
          this.datasets$.next(data);
          this.totalCount$.next(totalCount);
          this.updateVisiblePages(totalCount);
          this.isLoading$.next(false);
          this.cdr.markForCheck();
        },
        error: (error) => {
          console.error('Error loading datasets:', error);
          this.error$.next('Failed to load datasets. Please try again.');
          this.isLoading$.next(false);
          this.cdr.markForCheck();
        }
      });

    // Initialize the first load
    this.loadDatasets();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();

    // Clear BehaviorSubjects to prevent memory leaks
    this.datasets$.complete();
    this.isLoading$.complete();
    this.error$.complete();
    this.totalCount$.complete();
    this.filterParams$.complete();
  }

  /**
   * Set up listeners for filter form controls to update filter params
   */
  setupFilterListeners(): void {
    // Search with debounce
    this.filterForm.get('search')?.valueChanges
      .pipe(
        debounceTime(400),
        distinctUntilChanged(),
        takeUntil(this.destroy$)
      )
      .subscribe((value: string) => {
        this.updateFilterParams({ search: value, page: 1 });
      });

    // Status filter
    this.filterForm.get('status')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.updateFilterParams({
          status: value || undefined,
          page: 1
        });
      });

    // Type filter
    this.filterForm.get('type')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.updateFilterParams({
          type: value || undefined,
          page: 1
        });
      });

    // Visibility filter
    this.filterForm.get('isPublic')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        let isPublic: boolean | undefined;

        if (value === 'true') {
          isPublic = true;
        } else if (value === 'false') {
          isPublic = false;
        } else {
          isPublic = undefined;
        }

        this.updateFilterParams({
          is_public: isPublic,
          page: 1
        });
      });

    // Date range filter
    this.filterForm.get('dateRange')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        const dateFilter: Partial<DatasetFilterParams> = { page: 1 };

        // Map date range options to actual date values
        if (value) {
          const now = new Date();
          const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

          switch (value) {
            case 'today':
              dateFilter.dateFrom = today.toISOString();
              break;
            case 'yesterday':
              const yesterday = new Date(today);
              yesterday.setDate(yesterday.getDate() - 1);
              dateFilter.dateFrom = yesterday.toISOString();
              dateFilter.dateTo = today.toISOString();
              break;
            case 'week':
              const weekStart = new Date(today);
              weekStart.setDate(weekStart.getDate() - weekStart.getDay());
              dateFilter.dateFrom = weekStart.toISOString();
              break;
            case 'month':
              const monthStart = new Date(today.getFullYear(), today.getMonth(), 1);
              dateFilter.dateFrom = monthStart.toISOString();
              break;
            // 'custom' and other values handled in the UI with a date picker
          }
        } else {
          dateFilter.dateFrom = undefined;
          dateFilter.dateTo = undefined;
        }

        this.updateFilterParams(dateFilter);
      });

    // Size range filter
    this.filterForm.get('sizeRange')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        const sizeFilter: Partial<DatasetFilterParams> = { page: 1 };

        // Map size options to byte ranges
        if (value) {
          switch (value) {
            case 'small':
              sizeFilter.sizeMax = 1024 * 1024; // 1MB
              break;
            case 'medium':
              sizeFilter.sizeMin = 1024 * 1024; // 1MB
              sizeFilter.sizeMax = 10 * 1024 * 1024; // 10MB
              break;
            case 'large':
              sizeFilter.sizeMin = 10 * 1024 * 1024; // 10MB
              break;
          }
        } else {
          sizeFilter.sizeMin = undefined;
          sizeFilter.sizeMax = undefined;
        }

        this.updateFilterParams(sizeFilter);
      });
  }

  /**
   * Update filter parameters and trigger reload
   */
  private updateFilterParams(params: Partial<DatasetFilterParams>): void {
    this.filterParams$.next({
      ...this.filterParams$.value,
      ...params
    });
  }

  /**
   * Load datasets from cache or API
   */
  private loadDatasetsFromCache(params: DatasetFilterParams): Observable<{ data: Dataset[], totalCount: number }> {
    const cacheKey = this.getCacheKey(params);
    const cached = this.cache.get(cacheKey);
    const now = Date.now();

    if (cached && now - cached.timestamp < this.cacheExpiryMs) {
      // Return cached data
      return of({
        data: cached.data,
        totalCount: cached.totalCount
      });
    }

    // No valid cache, load from API
    return this.fetchDatasetsFromApi(params);
  }

  /**
   * Create a cache key from filter parameters
   */
  private getCacheKey(params: DatasetFilterParams): string {
    return JSON.stringify(params);
  }

  /**
   * Fetch datasets from the API
   */
  private fetchDatasetsFromApi(params: DatasetFilterParams): Observable<{ data: Dataset[], totalCount: number }> {
    this.isLoading$.next(true);
    this.error$.next(null);

    return this.datasetService.getDatasets(params).pipe(
      tap(response => {
        // Store in cache
        this.cache.set(this.getCacheKey(params), {
          data: response.datasets,
          totalCount: response.totalCount,
          timestamp: Date.now()
        });
      }),
      map(response => ({
        data: response.datasets,
        totalCount: response.totalCount
      })),
      catchError(error => {
        this.alertService.showAlert({
          show: true,
          message: 'Failed to load datasets. Please try again.',
          title: 'Error'
        });
        console.error('Error fetching datasets:', error);
        throw error;
      }),
      finalize(() => {
        this.isLoading$.next(false);
      })
    );
  }

  /**
   * Trigger dataset loading
   */
  loadDatasets(): void {
    // Just update the current filter params to trigger the loading
    this.filterParams$.next({ ...this.filterParams$.value });
  }

  /**
   * Update the array of visible page numbers
   */
  updateVisiblePages(totalCount: number): void {
    const maxVisiblePages = 5;
    const totalPages = Math.ceil(totalCount / this.itemsPerPage);
    const currentPage = this.filterParams$.value.page || 1;
    const pages: number[] = [];

    if (totalPages <= maxVisiblePages) {
      // If total pages are less than max visible, show all pages
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Always show first page
      pages.push(1);

      let startPage = Math.max(2, currentPage - 1);
      let endPage = Math.min(totalPages - 1, currentPage + 1);

      // Adjust if we're near the start or end
      if (currentPage <= 3) {
        endPage = Math.min(totalPages - 1, 4);
      } else if (currentPage >= totalPages - 2) {
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
    this.cdr.markForCheck();
  }

  /**
   * Navigate to specified page
   */
  onPageChange(page: number, event?: Event): void {
    if (event) {
      event.preventDefault();
    }

    if (page < 1) return;

    this.updateFilterParams({ page });
  }

  /**
   * Go to previous page
   */
  goToPreviousPage(event?: Event): void {
    if (event) {
      event.preventDefault();
    }

    const currentPage = this.filterParams$.value.page || 1;
    if (currentPage > 1) {
      this.onPageChange(currentPage - 1);
    }
  }

  /**
   * Go to next page
   */
  goToNextPage(event?: Event): void {
    if (event) {
      event.preventDefault();
    }

    const currentPage = this.filterParams$.value.page || 1;
    const totalPages = Math.ceil((this.totalCount$.value || 0) / this.itemsPerPage);

    if (currentPage < totalPages) {
      this.onPageChange(currentPage + 1);
    }
  }

  /**
   * Check if we're on the first page
   */
  isFirstPage(): boolean {
    return (this.filterParams$.value.page || 1) <= 1;
  }

  /**
   * Check if we're on the last page
   */
  isLastPage(): boolean {
    const currentPage = this.filterParams$.value.page || 1;
    const totalCount = this.totalCount$.value || 0;
    const totalPages = Math.ceil(totalCount / this.itemsPerPage);

    return currentPage >= totalPages;
  }

  /**
   * Get current page number
   */
  getCurrentPage(): number {
    return this.filterParams$.value.page || 1;
  }

  /**
   * Get displayed item range
   */
  getDisplayedRange(): { start: number; end: number; total: number } {
    const currentPage = this.filterParams$.value.page || 1;
    const totalCount = this.totalCount$.value || 0;

    const start = (currentPage - 1) * this.itemsPerPage + 1;
    const end = Math.min(currentPage * this.itemsPerPage, totalCount);

    return { start, end, total: totalCount };
  }

  /**
   * Clear all filters
   */
  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      type: '',
      isPublic: 'true',
      dateRange: '',
      sizeRange: ''
    });

    // Reset filter params to defaults
    this.updateFilterParams({
      page: 1,
      limit: this.itemsPerPage,
      search: '',
      status: undefined,
      type: undefined,
      is_public: true,
      sortBy: 'createdAt',
      sortDirection: 'desc',
      dateFrom: undefined,
      dateTo: undefined,
      sizeMin: undefined,
      sizeMax: undefined
    });
  }

  /**
   * Change sort column/direction
   */
  onSortChange(sortBy: 'name' | 'createdAt' | 'updatedAt' | 'documentCount'): void {
    const currentParams = this.filterParams$.value;

    // Toggle direction if same sort field
    const sortDirection = currentParams.sortBy === sortBy && currentParams.sortDirection === 'asc'
      ? 'desc'
      : 'asc';

    this.updateFilterParams({ sortBy, sortDirection });
  }

  /**
   * Navigate to dataset detail page
   */
  onDatasetClick(dataset: Dataset): void {
    this.ngZone.run(() => {
      this.router.navigate(['app/datasets/datasets', dataset.id]);
    });
  }

  /**
   * Navigate to dataset edit page
   */
  onEditDataset(event: Event, datasetId: string): void {
    event.stopPropagation(); // Prevent row click
    this.ngZone.run(() => {
      this.router.navigate(['app/datasets/datasets', datasetId]);
    });
  }

  /**
   * Navigate to create new dataset page
   */
  createNewDataset(event: Event): void {
    event.preventDefault();
    this.ngZone.run(() => {
      this.router.navigate(['app/datasets/datasets/upload']);
    });
  }

  /**
   * Confirm and delete dataset
   */
  confirmDeleteDataset(event: Event, datasetId: string): void {
    event.stopPropagation(); // Prevent navigation to detail page

    if (confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
      this.deleteDataset(datasetId);
    }
  }

  /**
   * Delete dataset
   */
  private deleteDataset(datasetId: string): void {
    this.datasetService.deleteDataset(datasetId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.alertService.showAlert({
            show: true,
            message: 'Dataset deleted successfully',
            title: 'Success'
          });

          // Clear cache and reload
          this.clearCache();
          this.loadDatasets();
        },
        error: (error) => {
          this.alertService.showAlert({
            show: true,
            message: 'Failed to delete dataset. Please try again.',
            title: 'Error'
          });
          console.error('Error deleting dataset:', error);
        }
      });
  }

  /**
   * Clear the cache
   */
  private clearCache(): void {
    this.cache.clear();
  }

  /**
   * Format date for display
   */
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
   * Format file size in bytes to a human-readable string
   */
  formatFileSize(bytes: number | undefined): string {
    if (bytes === undefined || bytes === 0) return 'N/A';

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }

    return `${value.toFixed(1)} ${units[unitIndex]}`;
  }

  /**
   * Get CSS class for status badge
   */
  getStatusClass(status: string): string {
    switch (status.toLowerCase()) {
      case 'ready':
        return 'status-ready';
      case 'processing':
        return 'status-processing';
      case 'error':
        return 'status-error';
      default:
        return '';
    }
  }
}
