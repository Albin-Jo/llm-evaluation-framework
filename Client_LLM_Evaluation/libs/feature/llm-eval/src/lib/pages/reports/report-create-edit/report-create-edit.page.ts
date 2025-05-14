import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';

import {
  ReportCreate,
  Report,
  ReportUpdate,
  ReportFormat,
  ReportStatus,
  Evaluation,
  EvaluationStatus,
} from '@ngtx-apps/data-access/models';
import {
  ReportService,
  EvaluationService,
} from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-report-create-edit',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterModule],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './report-create-edit.page.html',
  styleUrls: ['./report-create-edit.page.scss'],
})
export class ReportCreateEditPage implements OnInit, OnDestroy {
  selectedTabIndex = 0;
  reportForm!: FormGroup;
  isEditMode = false;
  reportId: string | null = null;
  isLoading = false;
  isSaving = false;
  error: string | null = null;
  pageTitle = 'Create New Report';

  // Format options
  reportFormats = [
    { value: ReportFormat.PDF, label: 'PDF' },
    { value: ReportFormat.HTML, label: 'HTML' },
    { value: ReportFormat.JSON, label: 'JSON' },
  ];

  // Evaluation options for dropdown
  evaluations: Evaluation[] = [];

  // Expose enum to template
  ReportFormat = ReportFormat;

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private reportService: ReportService,
    private evaluationService: EvaluationService,
    private route: ActivatedRoute,
    private router: Router,
    private notificationService: NotificationService
  ) {}

  ngOnInit(): void {
    this.initializeForm();
    this.loadEvaluations();

    this.reportId = this.route.snapshot.paramMap.get('id');

    if (this.reportId) {
      this.isEditMode = true;
      this.pageTitle = 'Edit Report';
      this.loadReportData();
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Initialize the form with default values and validators
   */
  private initializeForm(): void {
    this.reportForm = this.fb.group({
      name: ['', [Validators.required, Validators.maxLength(255)]],
      description: [''],
      evaluation_id: ['', Validators.required],
      format: [ReportFormat.PDF, Validators.required],
      config: this.fb.group({
        include_executive_summary: [true],
        include_evaluation_details: [true],
        include_metrics_overview: [true],
        include_detailed_results: [true],
        include_agent_responses: [true],
        max_examples: [10, [Validators.min(1), Validators.max(100)]],
      }),
    });
  }

  /**
   * Load evaluations for dropdown
   */
  loadEvaluations(): void {
    this.isLoading = true;
    this.evaluationService
      .getEvaluations({
        status: EvaluationStatus.COMPLETED, // Only show completed evaluations
        limit: 100, // Get more evaluations to have a good selection
      })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.evaluations = response.evaluations;
          this.isLoading = false;
        },
        error: (err) => {
          this.error = 'Failed to load evaluations. Please try again.';
          this.isLoading = false;
          console.error('Error loading evaluations:', err);
        },
      });
  }

  /**
   * Load report data for edit mode
   */
  loadReportData(): void {
    if (!this.reportId) return;

    this.isLoading = true;
    this.error = null;

    this.reportService
      .getReport(this.reportId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (report) => {
          this.populateForm(report);
          this.isLoading = false;
        },
        error: (err) => {
          this.error = 'Failed to load report data. Please try again.';
          this.isLoading = false;
          console.error('Error loading report data:', err);
          this.notificationService.error(this.error);
        },
      });
  }

  /**
   * Populate form with existing report data
   */
  populateForm(report: Report): void {
    // Build config form group if needed
    if (report.config) {
      const configGroup = this.fb.group({
        include_executive_summary: [
          report.config['include_executive_summary'] ?? true,
        ],
        include_evaluation_details: [
          report.config['include_evaluation_details'] ?? true,
        ],
        include_metrics_overview: [
          report.config['include_metrics_overview'] ?? true,
        ],
        include_detailed_results: [
          report.config['include_detailed_results'] ?? true,
        ],
        include_agent_responses: [
          report.config['include_agent_responses'] ?? true,
        ],
        max_examples: [report.config['max_examples'] ?? 10],
      });

      // Replace the config group in the form
      this.reportForm.setControl('config', configGroup);
    }

    // Update form values
    this.reportForm.patchValue({
      name: report.name,
      description: report.description || '',
      evaluation_id: report.evaluation_id,
      format: report.format,
    });
  }

  /**
   * Handle form submission
   */
  saveReport(): void {
    if (this.reportForm.invalid) {
      this.markFormGroupTouched(this.reportForm);
      return;
    }

    this.isSaving = true;
    const formValue = this.reportForm.getRawValue();

    // Transform form values to match the API contract
    const reportData = {
      name: formValue.name,
      description: formValue.description,
      evaluation_id: formValue.evaluation_id,
      format: formValue.format,
      config: formValue.config,
      ...formValue.config, // Spread config values at root level for creation
    };

    if (this.isEditMode) {
      // For updates, don't spread config values
      const updateData: ReportUpdate = {
        name: formValue.name,
        description: formValue.description,
        format: formValue.format,
        config: formValue.config,
      };
      this.updateReport(updateData);
    } else {
      this.createReport(reportData as ReportCreate);
    }
  }

  /**
   * Create a new report
   */
  createReport(formData: ReportCreate): void {
    this.reportService
      .createReport(formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.isSaving = false;
          this.notificationService.success('Report created successfully');
          this.router.navigate(['app/reports', response.id]);
        },
        error: (err) => {
          this.isSaving = false;
          this.notificationService.error(
            'Failed to create report. Please try again.'
          );
          console.error('Error creating report:', err);
        },
      });
  }

  /**
   * Update an existing report
   */
  updateReport(formData: ReportUpdate): void {
    if (!this.reportId) return;

    this.reportService
      .updateReport(this.reportId, formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.isSaving = false;
          this.notificationService.success('Report updated successfully');
          this.router.navigate(['app/reports', response.id]);
        },
        error: (err) => {
          this.isSaving = false;
          this.notificationService.error(
            'Failed to update report. Please try again.'
          );
          console.error('Error updating report:', err);
        },
      });
  }

  /**
   * Check if a field is invalid and has been touched
   */
  isFieldInvalid(fieldName: string): boolean {
    const control = this.reportForm.get(fieldName);
    return !!control && control.invalid && (control.dirty || control.touched);
  }

  /**
   * Get error message for field validation
   */
  getErrorMessage(fieldName: string): string {
    const control = this.reportForm.get(fieldName);
    if (!control || !control.errors) return '';

    if (control.errors['required']) {
      return `${
        fieldName.charAt(0).toUpperCase() + fieldName.slice(1).replace('_', ' ')
      } is required`;
    }
    if (control.errors['maxlength']) {
      return `${
        fieldName.charAt(0).toUpperCase() + fieldName.slice(1)
      } cannot exceed ${control.errors['maxlength'].requiredLength} characters`;
    }
    if (control.errors['min']) {
      return `Value must be at least ${control.errors['min'].min}`;
    }
    if (control.errors['max']) {
      return `Value cannot exceed ${control.errors['max'].max}`;
    }

    return 'Invalid value';
  }

  /**
   * Helper method to mark all controls in a form group as touched
   */
  markFormGroupTouched(formGroup: FormGroup): void {
    Object.values(formGroup.controls).forEach((control) => {
      control.markAsTouched();

      if ((control as FormGroup).controls) {
        this.markFormGroupTouched(control as FormGroup);
      }
    });
  }

  /**
   * Navigate to the next tab
   */
  nextTab(): void {
    if (this.selectedTabIndex < 2) {
      this.selectedTabIndex++;
    }
  }

  /**
   * Navigate to the previous tab
   */
  previousTab(): void {
    if (this.selectedTabIndex > 0) {
      this.selectedTabIndex--;
    }
  }

  /**
   * Navigate back to the reports list
   */
  cancel(): void {
    this.router.navigate(['app/reports']);
  }

  /**
   * Check if basic info form is valid
   */
  isBasicInfoValid(): boolean {
    const nameValid = this.reportForm.get('name')?.valid === true;
    const evaluationValid =
      this.reportForm.get('evaluation_id')?.valid === true;

    return nameValid && evaluationValid;
  }

  /**
   * Set selected format
   */
  setFormat(format: ReportFormat): void {
    this.reportForm.get('format')?.setValue(format);
  }

  /**
   * Check if a specific format is selected
   */
  isFormatSelected(format: ReportFormat): boolean {
    return this.reportForm.get('format')?.value === format;
  }
}
