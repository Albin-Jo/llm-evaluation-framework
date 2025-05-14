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
import { Subject, takeUntil } from 'rxjs';
import { debounceTime, distinctUntilChanged, finalize } from 'rxjs/operators';
import {
  Report,
  ReportFilterParams,
  ReportStatus,
  ReportFormat,
} from '@ngtx-apps/data-access/models';
import { ReportService } from '@ngtx-apps/data-access/services';
import {
  QracTextBoxComponent,
  QracSelectComponent,
} from '@ngtx-apps/ui/components';
import {
  ConfirmationDialogService,
  NotificationService,
} from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-reports',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,

    QracTextBoxComponent,
    QracSelectComponent,
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './reports.page.html',
  styleUrls: ['./reports.page.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ReportsPage implements OnInit, OnDestroy {
  reports: Report[] = [];
  totalCount = 0;
  isLoading = false;
  error: string | null = null;
  currentPage = 1;
  itemsPerPage = 5;
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: ReportFilterParams = {
    page: 1,
    limit: 5,
    sortBy: 'created_at',
    sortDirection: 'desc',
  };

  // For filtering by status
  statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: ReportStatus.DRAFT, label: 'Draft' },
    { value: ReportStatus.GENERATED, label: 'Generated' },
  ];

  // Format options
  formatOptions = [
    { value: '', label: 'All Formats' },
    { value: ReportFormat.PDF, label: 'PDF' },
    { value: ReportFormat.HTML, label: 'HTML' },
    { value: ReportFormat.JSON, label: 'JSON' },
  ];

  // Expose enums for template usage
  ReportStatus = ReportStatus;
  ReportFormat = ReportFormat;

  private destroy$ = new Subject<void>();

  constructor(
    private reportService: ReportService,
    private confirmationDialogService: ConfirmationDialogService,
    private notificationService: NotificationService,
    private router: Router,
    private fb: FormBuilder,
    private cdr: ChangeDetectorRef // Added for change detection
  ) {
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      format: [''],
    });
  }

  ngOnInit(): void {
    this.setupFilterListeners();
    this.loadReports();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
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
        this.loadReports();
      });

    // Listen to status changes
    this.filterForm
      .get('status')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.status = value ? (value as ReportStatus) : undefined;
        this.filterParams.page = 1;
        this.loadReports();
      });

    // Listen to format changes
    this.filterForm
      .get('format')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.format = value ? (value as ReportFormat) : undefined;
        this.filterParams.page = 1;
        this.loadReports();
      });
  }

  loadReports(): void {
    this.isLoading = true;
    this.error = null;
    this.cdr.markForCheck(); // Mark for check at start of loading

    this.reportService
      .getReports(this.filterParams)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isLoading = false;
          this.cdr.markForCheck(); // Mark for check when loading completes
        })
      )
      .subscribe({
        next: (response) => {
          this.reports = response.reports;
          this.totalCount = response.totalCount;
          this.updateVisiblePages();
          this.cdr.markForCheck(); // Mark for check when data is updated
        },
        error: (error) => {
          this.error = 'Failed to load reports. Please try again.';
          this.notificationService.error(this.error);
          console.error('Error loading reports:', error);
          this.cdr.markForCheck(); // Mark for check on error
        },
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
    this.loadReports();
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      format: '',
    });

    this.filterParams.name = undefined;
    this.filterParams.status = undefined;
    this.filterParams.format = undefined;
    this.filterParams.page = 1;

    this.loadReports();
  }

  onSortChange(sortBy: string): void {
    const validSortFields = [
      'created_at',
      'updated_at',
      'name',
      'status',
      'format',
    ];

    if (validSortFields.includes(sortBy)) {
      if (this.filterParams.sortBy === sortBy) {
        this.filterParams.sortDirection =
          this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        this.filterParams.sortBy = sortBy as
          | 'created_at'
          | 'name'
          | 'updated_at';
        this.filterParams.sortDirection = 'desc';
      }

      this.filterParams.page = 1;
      this.loadReports();
    }
  }

  onReportClick(report: Report): void {
    this.router.navigate(['app/reports', report.id]);
  }

  onViewReport(event: Event, reportId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/reports', reportId]);
  }

  onEditReport(event: Event, reportId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/reports', reportId, 'edit']);
  }

  createNewReport(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/reports/create']);
  }

  confirmDeleteReport(event: Event, reportId: string): void {
    event.stopPropagation();

    this.confirmationDialogService
      .confirmDelete('Report')
      .subscribe((confirmed) => {
        if (confirmed) {
          this.deleteReport(reportId);
        }
      });
  }

  private deleteReport(reportId: string): void {
    this.reportService
      .deleteReport(reportId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.notificationService.success('Report deleted successfully');
          this.loadReports();
        },
        error: (error) => {
          this.notificationService.error(
            'Failed to delete report. Please try again.'
          );
          console.error('Error deleting report:', error);
        },
      });
  }

  generateReport(event: Event, reportId: string): void {
    event.stopPropagation();

    this.confirmationDialogService
      .confirm({
        title: 'Generate Report',
        message: 'Are you sure you want to generate this report?',
        confirmText: 'Generate',
        cancelText: 'Cancel',
        type: 'info',
      })
      .subscribe((confirmed) => {
        if (confirmed) {
          this.reportService
            .generateReport(reportId)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.notificationService.success(
                  'Report generated successfully'
                );
                this.loadReports();
              },
              error: (error) => {
                this.notificationService.error(
                  'Failed to generate report. Please try again.'
                );
                console.error('Error generating report:', error);
              },
            });
        }
      });
  }

  downloadReport(event: Event, reportId: string): void {
    event.stopPropagation();

    const report = this.reports.find((r) => r.id === reportId);
    if (!report) return;

    this.isLoading = true;
    this.cdr.markForCheck();

    this.reportService
      .downloadReport(reportId)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isLoading = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe({
        next: (blob) => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `report-${reportId}.${report.format.toLowerCase()}`;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);

          this.notificationService.success('Report downloaded successfully');
        },
        error: (error) => {
          this.notificationService.error(
            'Failed to download report. Please try again.'
          );
          console.error('Error downloading report:', error);
        },
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

  /**
   * Truncate text to specified length
   */
  truncateText(text: string, maxLength = 100): string {
    if (!text) return '';
    return text.length > maxLength
      ? `${text.substring(0, maxLength)}...`
      : text;
  }

  /**
   * Get status badge class based on report status
   */
  getStatusBadgeClass(status: ReportStatus): string {
    switch (status) {
      case ReportStatus.GENERATED:
        return 'generated';
      case ReportStatus.DRAFT:
        return 'draft';
      default:
        return '';
    }
  }

  /**
   * Check if report can be generated (only DRAFT reports)
   */
  canGenerateReport(status: ReportStatus): boolean {
    return status === ReportStatus.DRAFT;
  }

  /**
   * Check if report can be downloaded (only GENERATED reports)
   */
  canDownloadReport(status: ReportStatus): boolean {
    return status === ReportStatus.GENERATED;
  }

  /**
   * Track by function for ngFor optimization
   */
  trackByReportId(index: number, report: Report): string {
    return report.id;
  }
}
