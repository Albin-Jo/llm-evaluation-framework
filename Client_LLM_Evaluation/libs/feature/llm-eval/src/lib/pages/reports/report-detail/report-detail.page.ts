/* Path: libs/feature/llm-eval/src/lib/pages/reports/report-detail/report-detail.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { finalize } from 'rxjs/operators';

import {
  Report,
  ReportStatus,
  ReportFormat,
  EvaluationDetail,
  EvaluationStatus
} from '@ngtx-apps/data-access/models';
import {
  ReportService,
  EvaluationService
} from '@ngtx-apps/data-access/services';
import {
  ConfirmationDialogService,
  NotificationService
} from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-report-detail',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './report-detail.page.html',
  styleUrls: ['./report-detail.page.scss']
})
export class ReportDetailPage implements OnInit, OnDestroy {
  report: Report | null = null;
  evaluation: EvaluationDetail | null = null;
  reportId: string = '';
  isLoading = false;
  error: string | null = null;
  isDownloading = false;

  // Expose enums for template usage
  ReportStatus = ReportStatus;
  ReportFormat = ReportFormat;

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private reportService: ReportService,
    private evaluationService: EvaluationService,
    private confirmationDialogService: ConfirmationDialogService,
    private notificationService: NotificationService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.pipe(
      takeUntil(this.destroy$)
    ).subscribe(params => {
      const id = params.get('id');
      if (id) {
        this.reportId = id;
        this.loadReport(id);
      } else {
        this.error = 'Report ID not found.';
      }
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadReport(id: string): void {
    this.isLoading = true;
    this.error = null;

    this.reportService.getReport(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (report) => {
          this.report = report;
          this.isLoading = false;

          // Load evaluation data if we have evaluation_id
          if (report.evaluation_id) {
            this.loadEvaluationDetails(report.evaluation_id);
          }
        },
        error: (error) => {
          this.error = 'Failed to load report details. Please try again.';
          this.isLoading = false;
          console.error('Error loading report:', error);
        }
      });
  }

  loadEvaluationDetails(evaluationId: string): void {
    this.evaluationService.getEvaluation(evaluationId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (evaluation) => {
          this.evaluation = evaluation;
        },
        error: (error) => {
          console.error('Error loading evaluation details:', error);
          // Don't set main error - this is supplementary data
        }
      });
  }

  editReport(): void {
    if (this.report) {
      this.router.navigate(['app/reports', this.report.id, 'edit']);
    }
  }

  deleteReport(): void {
    if (!this.report) return;

    this.confirmationDialogService.confirmDelete('Report')
      .subscribe(confirmed => {
        if (confirmed) {
          this.reportService.deleteReport(this.report!.id)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.notificationService.success('Report deleted successfully');
                // Navigate back to reports list
                this.router.navigate(['app/reports']);
              },
              error: (error) => {
                this.notificationService.error('Failed to delete report. Please try again.');
                console.error('Error deleting report:', error);
              }
            });
        }
      });
  }

  downloadReport(): void {
    if (!this.report) return;

    this.isDownloading = true;
    this.reportService.downloadReport(this.report.id)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => this.isDownloading = false)
      )
      .subscribe({
        next: (blob) => {
          // Create a URL for the blob
          const url = window.URL.createObjectURL(blob);
          // Create an anchor element and trigger download
          const a = document.createElement('a');
          a.href = url;
          a.download = `report-${this.report!.id}.${this.report!.format.toLowerCase()}`;
          document.body.appendChild(a);
          a.click();
          // Cleanup
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);

          this.notificationService.success('Report downloaded successfully');
        },
        error: (error) => {
          this.notificationService.error('Failed to download report. Please try again.');
          console.error('Error downloading report:', error);
        }
      });
  }

  generateReport(): void {
    if (!this.report) return;

    this.confirmationDialogService.confirm({
      title: 'Generate Report',
      message: 'Are you sure you want to generate this report?',
      confirmText: 'Generate',
      cancelText: 'Cancel',
      type: 'info'
    }).subscribe(confirmed => {
      if (confirmed) {
        this.isLoading = true;
        this.reportService.generateReport(this.report!.id)
          .pipe(
            takeUntil(this.destroy$),
            finalize(() => this.isLoading = false)
          )
          .subscribe({
            next: (response) => {
              this.report = response;
              this.notificationService.success('Report generated successfully');
            },
            error: (error) => {
              this.notificationService.error('Failed to generate report. Please try again.');
              console.error('Error generating report:', error);
            }
          });
      }
    });
  }

  viewFullReport(): void {
    if (this.report) {
      this.router.navigate(['app/reports', this.report.id, 'preview']);
    }
  }

  viewEvaluation(): void {
    if (this.report?.evaluation_id) {
      this.router.navigate(['app/evaluations', this.report.evaluation_id]);
    }
  }

  formatDate(dateString: string | undefined): string {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    } catch (e) {
      return 'Invalid date';
    }
  }

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

  canGenerateReport(): boolean {
    return this.report?.status === ReportStatus.DRAFT;
  }

  canDownloadReport(): boolean {
    return this.report?.status === ReportStatus.GENERATED;
  }

  getFormattedConfig(): string {
    if (!this.report?.config) return 'No configuration';
    return JSON.stringify(this.report.config, null, 2);
  }

  getEvaluationStatusClass(): string {
    if (!this.evaluation?.status) return '';
    return this.evaluation.status.toLowerCase();
  }
}
