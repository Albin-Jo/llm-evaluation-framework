/* Path: libs/feature/llm-eval/src/lib/pages/reports/report-preview/report-preview.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import {
  Report,
  ReportStatus
} from '@ngtx-apps/data-access/models';
import { ReportService } from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-report-preview',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './report-preview.page.html',
  styleUrls: ['./report-preview.page.scss']
})
export class ReportPreviewPage implements OnInit, OnDestroy {
  report: Report | null = null;
  reportId: string = '';
  isLoading = false;
  isDownloading = false;
  error: string | null = null;
  previewHtml: SafeHtml | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private reportService: ReportService,
    private sanitizer: DomSanitizer,
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

          // Check if report is generated
          if (report.status !== ReportStatus.GENERATED) {
            this.error = 'This report has not been generated yet. Please generate the report first.';
            this.isLoading = false;
            return;
          }

          // Load the preview
          this.loadPreview();
        },
        error: (error) => {
          this.error = 'Failed to load report details. Please try again.';
          this.isLoading = false;
          console.error('Error loading report:', error);
        }
      });
  }

  loadPreview(): void {
    if (!this.reportId) return;

    this.reportService.previewReport(this.reportId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (htmlContent) => {
          // Sanitize the HTML content
          this.previewHtml = this.sanitizer.bypassSecurityTrustHtml(htmlContent);
          this.isLoading = false;
        },
        error: (error) => {
          this.error = 'Failed to load report preview. Please try again.';
          this.isLoading = false;
          console.error('Error loading preview:', error);
        }
      });
  }

  downloadReport(): void {
    if (!this.report) return;

    this.isDownloading = true;
    this.reportService.downloadReport(this.report.id)
      .pipe(takeUntil(this.destroy$))
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

          this.isDownloading = false;
          this.notificationService.success('Report downloaded successfully');
        },
        error: (error) => {
          this.isDownloading = false;
          this.notificationService.error('Failed to download report. Please try again.');
          console.error('Error downloading report:', error);
        }
      });
  }

  sendReport(): void {
    if (!this.report) return;

    // Navigate to send report page or open a modal
    // For now, just show a notification that this feature is coming
    this.notificationService.info('Send report feature will be implemented soon.');
  }

  backToReport(): void {
    if (this.report) {
      this.router.navigate(['app/reports', this.report.id]);
    } else {
      this.router.navigate(['app/reports']);
    }
  }
}
