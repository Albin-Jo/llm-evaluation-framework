/* Path: libs/feature/llm-eval/src/lib/pages/datasets/datasets.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';

import {
  Dataset,
  DatasetFilterParams,
  DatasetStatus
} from '@ngtx-apps/data-access/models';
import { DatasetService } from '@ngtx-apps/data-access/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent
} from '@ngtx-apps/ui/components';
import { AlertService, ConfirmationDialogService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-datasets',
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
  templateUrl: './datasets.page.html',
  styleUrls: ['./datasets.page.scss']
})
export class DatasetsPage implements OnInit, OnDestroy {
  datasets: Dataset[] = [];
  totalCount = 0;
  isLoading = false;
  error: string | null = null;
  currentPage = 1;
  itemsPerPage = 10; // Standardized to 10
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: DatasetFilterParams = {
    page: 1,
    limit: 10, // Standardized to 10
    sortBy: 'createdAt',
    sortDirection: 'desc',
    is_public: true
  };

  // Options for filters
  statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: DatasetStatus.READY, label: 'Ready' },
    { value: DatasetStatus.PROCESSING, label: 'Processing' },
    { value: DatasetStatus.ERROR, label: 'Error' }
  ];

  formatOptions = [
    { value: '', label: 'All Types' },
    { value: 'csv', label: 'CSV' },
    { value: 'jsonl', label: 'JSONL' },
    { value: 'txt', label: 'Text' },
    { value: 'custom', label: 'Custom' }
  ];

  visibilityOptions = [
    { value: 'true', label: 'Public' },
    { value: 'false', label: 'Private' },
    { value: '', label: 'All' }
  ];

  dateRangeOptions = [
    { value: '', label: 'Any Time' },
    { value: 'today', label: 'Today' },
    { value: 'yesterday', label: 'Yesterday' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' },
    { value: 'custom', label: 'Custom Range' }
  ];

  sizeRangeOptions = [
    { value: '', label: 'Any Size' },
    { value: 'small', label: 'Small (<1MB)' },
    { value: 'medium', label: 'Medium (1-10MB)' },
    { value: 'large', label: 'Large (>10MB)' }
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private datasetService: DatasetService,
    private alertService: AlertService,
    private confirmationDialogService: ConfirmationDialogService,
    private router: Router,
    private fb: FormBuilder
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
    this.loadDatasets();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  setupFilterListeners(): void {
    // Search with debounce
    this.filterForm.get('search')?.valueChanges
      .pipe(
        debounceTime(400),
        distinctUntilChanged(),
        takeUntil(this.destroy$)
      )
      .subscribe((value: string) => {
        this.filterParams.search = value || undefined;
        this.filterParams.page = 1;
        this.loadDatasets();
      });

    // Status filter
    this.filterForm.get('status')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.status = value || undefined;
        this.filterParams.page = 1;
        this.loadDatasets();
      });

    // Type filter
    this.filterForm.get('type')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.type = value || undefined;
        this.filterParams.page = 1;
        this.loadDatasets();
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

        this.filterParams.is_public = isPublic;
        this.filterParams.page = 1;
        this.loadDatasets();
      });

    // Date range filter
    this.filterForm.get('dateRange')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.updateDateRangeFilter(value);
      });

    // Size range filter
    this.filterForm.get('sizeRange')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.updateSizeRangeFilter(value);
      });
  }

  private updateDateRangeFilter(value: string): void {
    const dateFilter: Partial<DatasetFilterParams> = { page: 1 };

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
      }
    } else {
      dateFilter.dateFrom = undefined;
      dateFilter.dateTo = undefined;
    }

    this.filterParams = { ...this.filterParams, ...dateFilter };
    this.loadDatasets();
  }

  private updateSizeRangeFilter(value: string): void {
    const sizeFilter: Partial<DatasetFilterParams> = { page: 1 };

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

    this.filterParams = { ...this.filterParams, ...sizeFilter };
    this.loadDatasets();
  }

  loadDatasets(): void {
    this.isLoading = true;
    this.error = null;

    this.datasetService.getDatasets(this.filterParams)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.datasets = response.datasets;
          this.totalCount = response.totalCount;
          this.isLoading = false;
          this.updateVisiblePages();
        },
        error: (error) => {
          this.error = 'Failed to load datasets. Please try again.';
          this.alertService.showAlert({
            show: true,
            message: this.error,
            title: 'Error'
          });
          this.isLoading = false;
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
    this.loadDatasets();
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      type: '',
      isPublic: 'true',
      dateRange: '',
      sizeRange: ''
    });

    this.filterParams = {
      page: 1,
      limit: this.itemsPerPage,
      search: undefined,
      status: undefined,
      type: undefined,
      is_public: true,
      sortBy: 'createdAt',
      sortDirection: 'desc',
      dateFrom: undefined,
      dateTo: undefined,
      sizeMin: undefined,
      sizeMax: undefined
    };

    this.loadDatasets();
  }

  onSortChange(sortBy: 'name' | 'createdAt' | 'updatedAt' | 'documentCount'): void {
    if (this.filterParams.sortBy === sortBy) {
      this.filterParams.sortDirection =
        this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.filterParams.sortBy = sortBy;
      this.filterParams.sortDirection = 'desc';
    }

    this.loadDatasets();
  }

  onDatasetClick(dataset: Dataset): void {
    this.router.navigate(['app/datasets/datasets', dataset.id]);
  }

  onEditDataset(event: Event, datasetId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/datasets/datasets', datasetId]);
  }

  createNewDataset(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/datasets/datasets/upload']);
  }

  confirmDeleteDataset(event: Event, datasetId: string): void {
    event.stopPropagation();

    this.confirmationDialogService.confirmDelete('Dataset')
      .subscribe(confirmed => {
        if (confirmed) {
          this.deleteDataset(datasetId);
        }
      });
  }

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
          this.loadDatasets();
        },
        error: (error) => {
          this.alertService.showAlert({
            show: true,
            message: 'Failed to delete dataset. Please try again.',
            title: 'Error'
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