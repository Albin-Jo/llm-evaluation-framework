import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, takeUntil, forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';

import {
  ComparisonCreate,
  Comparison,
  ComparisonUpdate,
  Evaluation,
  EvaluationStatus,
} from '@ngtx-apps/data-access/models';
import {
  ComparisonService,
  EvaluationService,
} from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-comparison-create-edit',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterModule],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './comparison-create-edit.page.html',
  styleUrls: ['./comparison-create-edit.page.scss'],
})
export class ComparisonCreateEditPage implements OnInit, OnDestroy {
  selectedTabIndex = 0;
  comparisonForm!: FormGroup;
  isEditMode = false;
  comparisonId: string | null = null;
  isLoading = false;
  isSaving = false;
  error: string | null = null;
  pageTitle = 'Create New Comparison';

  // Evaluations for dropdown
  evaluations: Evaluation[] = [];

  // Cache for evaluation names
  evaluationNameCache: Record<string, string> = {};

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private comparisonService: ComparisonService,
    private evaluationService: EvaluationService,
    private route: ActivatedRoute,
    private router: Router,
    private notificationService: NotificationService
  ) {}

  ngOnInit(): void {
    this.initializeForm();
    this.loadEvaluations();

    this.comparisonId = this.route.snapshot.paramMap.get('id');

    if (this.comparisonId) {
      this.isEditMode = true;
      this.pageTitle = 'Edit Comparison';
      this.loadComparisonData();
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
    this.comparisonForm = this.fb.group({
      name: ['', [Validators.required, Validators.maxLength(255)]],
      description: [''],
      evaluation_a_id: ['', Validators.required],
      evaluation_b_id: ['', Validators.required],
      config: this.fb.group({
        threshold: [0.05, [Validators.required, Validators.min(0.01), Validators.max(0.25)]],
        normalize_scores: [true],
        detailed_analysis: [false],
      }),
    });
  }

  /**
   * Load evaluations for the dropdowns
   */
  loadEvaluations(): void {
    this.isLoading = true;
    this.error = null;

    // Load only completed evaluations
    this.evaluationService
      .getEvaluations({
        status: EvaluationStatus.COMPLETED,
        limit: 100, // Get more evaluations to have a good selection
      })
      .pipe(
        takeUntil(this.destroy$),
        catchError((error) => {
          this.error = 'Failed to load evaluations. Please try again.';
          console.error('Error loading evaluations:', error);
          return of({ evaluations: [], totalCount: 0 });
        })
      )
      .subscribe((response) => {
        this.evaluations = response.evaluations;
        this.evaluations.forEach(evaluation => {
          this.evaluationNameCache[evaluation.id] = evaluation.name;
        });
        this.isLoading = false;
      });
  }

  /**
   * Load comparison data for edit mode
   */
  loadComparisonData(): void {
    if (!this.comparisonId) return;

    this.isLoading = true;
    this.error = null;

    this.comparisonService
      .getComparison(this.comparisonId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (comparison) => {
          this.populateForm(comparison);
          this.isLoading = false;
        },
        error: (err) => {
          this.error = 'Failed to load comparison data. Please try again.';
          this.isLoading = false;
          console.error('Error loading comparison data:', err);
          this.notificationService.error(this.error);
        },
      });
  }

  /**
   * Populate form with existing comparison data
   */
  populateForm(comparison: Comparison): void {
    // Build config form group if needed
    if (comparison.config) {
      const configGroup = this.fb.group({
        threshold: [
          comparison.config['threshold'] ?? 0.05,
          [Validators.required, Validators.min(0.01), Validators.max(0.25)],
        ],
        normalize_scores: [comparison.config['normalize_scores'] ?? true],
        detailed_analysis: [comparison.config['detailed_analysis'] ?? false],
      });

      // Replace the config group in the form
      this.comparisonForm.setControl('config', configGroup);
    }

    // Update form values
    this.comparisonForm.patchValue({
      name: comparison.name,
      description: comparison.description || '',
      evaluation_a_id: comparison.evaluation_a_id,
      evaluation_b_id: comparison.evaluation_b_id,
    });
  }

  /**
   * Handle form submission
   */
  saveComparison(): void {
    if (this.comparisonForm.invalid) {
      this.markFormGroupTouched(this.comparisonForm);
      return;
    }

    this.isSaving = true;
    const formValue = this.comparisonForm.getRawValue();

    if (this.isEditMode) {
      // For updates, don't include evaluation IDs which can't be changed
      const updateData: ComparisonUpdate = {
        name: formValue.name,
        description: formValue.description,
        config: formValue.config,
      };
      this.updateComparison(updateData);
    } else {
      this.createComparison(formValue as ComparisonCreate);
    }
  }

  /**
   * Create a new comparison
   */
  createComparison(formData: ComparisonCreate): void {
    this.comparisonService
      .createComparison(formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.isSaving = false;
          this.notificationService.success('Comparison created successfully');
          this.router.navigate(['app/comparisons', response.id]);
        },
        error: (err) => {
          this.isSaving = false;
          this.notificationService.error(
            'Failed to create comparison. Please try again.'
          );
          console.error('Error creating comparison:', err);
        },
      });
  }

  /**
   * Update an existing comparison
   */
  updateComparison(formData: ComparisonUpdate): void {
    if (!this.comparisonId) return;

    this.comparisonService
      .updateComparison(this.comparisonId, formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.isSaving = false;
          this.notificationService.success('Comparison updated successfully');
          this.router.navigate(['app/comparisons', response.id]);
        },
        error: (err) => {
          this.isSaving = false;
          this.notificationService.error(
            'Failed to update comparison. Please try again.'
          );
          console.error('Error updating comparison:', err);
        },
      });
  }

  /**
   * Check if a field is invalid and has been touched
   */
  isFieldInvalid(fieldName: string): boolean {
    const control = this.comparisonForm.get(fieldName);
    return !!control && control.invalid && (control.dirty || control.touched);
  }

  /**
   * Get error message for field validation
   */
  getErrorMessage(fieldName: string): string {
    const control = this.comparisonForm.get(fieldName);
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
   * Navigate back to the comparisons list
   */
  cancel(): void {
    this.router.navigate(['app/comparisons']);
  }

  /**
   * Check if basic info form is valid
   */
  isBasicInfoValid(): boolean {
    const nameValid = this.comparisonForm.get('name')?.valid === true;
    const evalAValid = this.comparisonForm.get('evaluation_a_id')?.valid === true;
    const evalBValid = this.comparisonForm.get('evaluation_b_id')?.valid === true;

    return nameValid && evalAValid && evalBValid;
  }

  /**
   * Get evaluation name by ID
   */
  getEvaluationName(id: string): string {
    if (!id) return '';

    // First check the cache
    if (this.evaluationNameCache[id]) {
      return this.evaluationNameCache[id];
    }

    // Then check the loaded evaluations
    const evaluation = this.evaluations.find(e => e.id === id);
    if (evaluation) {
      // Cache the name for future use
      this.evaluationNameCache[id] = evaluation.name;
      return evaluation.name;
    }

    // Return a placeholder if not found
    return id.substring(0, 8) + '...';
  }
}