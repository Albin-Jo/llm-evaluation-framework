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
    QracButtonComponent
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
  itemsPerPage = 10;
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: DatasetFilterParams = {
    page: 1,
    limit: 10,
    sortBy: 'createdAt',
    sortDirection: 'desc',
    is_public: true,
  };

  // For filtering by status
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

  private destroy$ = new Subject<void>();

  constructor(
    private datasetService: DatasetService,
    private alertService: AlertService,
    private router: Router,
    private fb: FormBuilder
  ) {
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      type: [''],
      isPublic: ['true']
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
    // Set up search debounce
    this.filterForm.get('search')?.valueChanges
      .pipe(
        debounceTime(400),
        distinctUntilChanged(),
        takeUntil(this.destroy$)
      )
      .subscribe((value: string) => {
        this.filterParams.search = value;
        this.filterParams.page = 1;
        this.loadDatasets();
      });

    // Listen to status changes
    this.filterForm.get('status')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        if (value) {
          this.filterParams.status = value as DatasetStatus;
        } else {
          this.filterParams.status = undefined;
        }
        this.filterParams.page = 1;
        this.loadDatasets();
      });

    // Listen to type changes
    this.filterForm.get('type')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        // Store the type value locally as it's not part of DatasetFilterParams
        // We'll handle it in the UI only, since the API doesn't support filtering by type
        this.filterParams.page = 1;
        this.loadDatasets();
      });

    // Listen to visibility changes
    this.filterForm.get('isPublic')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        if (value === 'true') {
          this.filterParams.is_public = true;
        } else if (value === 'false') {
          this.filterParams.is_public = false;
        } else {
          this.filterParams.is_public = undefined;
        }
        this.filterParams.page = 1;
        this.loadDatasets();
      });
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

          // Calculate pagination
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
          console.error('Error loading datasets:', error);
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
      isPublic: 'true'
    });

    // Reset filter params manually
    this.filterParams.search = '';
    this.filterParams.status = undefined;
    this.filterParams.is_public = true;
    this.filterParams.page = 1;

    this.loadDatasets();
  }

  onSortChange(sortBy: string): void {
    if (this.filterParams.sortBy === sortBy) {
      // Toggle direction if same sort field
      this.filterParams.sortDirection =
        this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      // Default to desc for new sort field
      this.filterParams.sortBy = sortBy as "createdAt" | "name" | "documentCount" | "updatedAt";
      this.filterParams.sortDirection = 'desc';
    }

    this.loadDatasets();
  }

  onDatasetClick(dataset: Dataset): void {
    this.router.navigate(['app/datasets/datasets', dataset.id]);
  }

  onEditDataset(event: Event, datasetId: string): void {
    event.stopPropagation(); // Prevent row click
    this.router.navigate(['app/datasets/datasets', datasetId, 'edit']);
  }

  createNewDataset(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/datasets/datasets/upload']);
  }

  confirmDeleteDataset(event: Event, datasetId: string): void {
    event.stopPropagation(); // Prevent navigation to detail page

    if (confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
      this.deleteDataset(datasetId);
    }
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
          this.loadDatasets(); // Reload the list
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
}
