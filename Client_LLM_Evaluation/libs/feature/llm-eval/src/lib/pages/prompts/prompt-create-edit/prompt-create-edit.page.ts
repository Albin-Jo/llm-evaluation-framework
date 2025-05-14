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

import { PromptService } from '@ngtx-apps/data-access/services';
import {
  PromptCreate,
  PromptResponse,
  PromptUpdate,
} from '@ngtx-apps/data-access/models';
import { AlertService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-prompt-create-edit',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterModule],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './prompt-create-edit.page.html',
  styleUrls: ['./prompt-create-edit.page.scss'],
})
export class PromptCreateEditPage implements OnInit, OnDestroy {
  promptForm!: FormGroup;
  isEditMode = false;
  promptId: string | null = null;
  isLoading = false;
  isSaving = false;
  error: string | null = null;
  pageTitle = 'Create New Prompt';
  parameterKeys: string[] = [];

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private promptService: PromptService,
    private route: ActivatedRoute,
    private router: Router,
    private alertService: AlertService
  ) {}

  ngOnInit(): void {
    this.initializeForm();
    this.promptId = this.route.snapshot.paramMap.get('id');

    if (this.promptId) {
      this.isEditMode = true;
      this.pageTitle = 'Edit Prompt';
      this.loadPromptData();
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
    this.promptForm = this.fb.group({
      name: ['', [Validators.required, Validators.maxLength(255)]],
      description: [''],
      content: ['', Validators.required],
      parameters: this.fb.group({}),
      version: ['1.0.0'],
      is_public: [false],
      template_id: [null],
    });

    // Clear parameter keys for a fresh form
    this.parameterKeys = [];
  }

  /**
   * Load prompt data for edit mode
   */
  loadPromptData(): void {
    if (!this.promptId) return;

    this.isLoading = true;
    this.error = null;

    this.promptService
      .getPromptById(this.promptId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (prompt: PromptResponse) => {
          this.populateForm(prompt);
          this.isLoading = false;
        },
        error: (err: Error) => {
          this.error = 'Failed to load prompt data. Please try again.';
          this.isLoading = false;
          console.error('Error loading prompt data:', err);
        },
      });
  }

  /**
   * Populate form with existing prompt data
   */
  populateForm(prompt: PromptResponse): void {
    this.promptForm.patchValue({
      name: prompt.name,
      description: prompt.description || '',
      content: prompt.content,
      version: prompt.version,
      is_public: prompt.is_public,
      template_id: prompt.template_id,
    });

    // Handle parameters if they exist
    if (prompt.parameters) {
      // Create dynamic form controls for each parameter
      const paramControls = this.fb.group({});
      this.parameterKeys = [];

      // Transform parameters object into form groups
      Object.entries(prompt.parameters).forEach(([key, value], index) => {
        const paramKey = `param${index + 1}`;

        // Create a form group for the parameter with name and value fields
        const paramGroup = this.fb.group({
          name: [key, Validators.required],
          value: [value],
        });

        paramControls.addControl(paramKey, paramGroup);
        this.parameterKeys.push(paramKey);
      });

      this.promptForm.setControl('parameters', paramControls);
    } else {
      this.parameterKeys = [];
    }
  }

  /**
   * Save the prompt (create or update)
   */
  savePrompt(): void {
    if (this.promptForm.invalid) {
      this.markFormGroupTouched(this.promptForm);
      return;
    }

    this.isSaving = true;
    const rawFormValue = this.promptForm.getRawValue();

    // Transform parameters from form structure to expected API structure
    const parameters: Record<string, string> = {};
    if (rawFormValue.parameters) {
      Object.values(rawFormValue.parameters).forEach((param: any) => {
        if (param.name) {
          parameters[param.name] = param.value || '';
        }
      });
    }

    // Create the final form data with transformed parameters
    const formValue = {
      ...rawFormValue,
      parameters,
    };

    if (this.isEditMode) {
      this.updatePrompt(formValue);
    } else {
      this.createPrompt(formValue);
    }
  }

  /**
   * Create a new prompt
   */
  createPrompt(formData: PromptCreate): void {
    this.promptService
      .createPrompt(formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: PromptResponse) => {
          this.isSaving = false;
          this.alertService.showAlert({
            show: true,
            message: 'Prompt created successfully',
            title: 'Success',
          });
          this.router.navigate(['app/prompts']);
        },
        error: (err: Error) => {
          this.isSaving = false;
          this.alertService.showAlert({
            show: true,
            message: 'Failed to create prompt. Please try again.',
            title: 'Error',
          });
          console.error('Error creating prompt:', err);
        },
      });
  }

  /**
   * Update an existing prompt
   */
  updatePrompt(formData: PromptUpdate): void {
    if (!this.promptId) return;

    this.promptService
      .updatePrompt(this.promptId, formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: PromptResponse) => {
          this.isSaving = false;
          this.alertService.showAlert({
            show: true,
            message: 'Prompt updated successfully',
            title: 'Success',
          });
          this.router.navigate(['app/prompts']);
        },
        error: (err: Error) => {
          this.isSaving = false;
          this.alertService.showAlert({
            show: true,
            message: 'Failed to update prompt. Please try again.',
            title: 'Error',
          });
          console.error('Error updating prompt:', err);
        },
      });
  }

  /**
   * Check if a field is invalid and has been touched
   */
  isFieldInvalid(fieldName: string): boolean {
    const control = this.promptForm.get(fieldName);
    return !!control && control.invalid && (control.dirty || control.touched);
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
   * Add a new parameter field
   */
  addParameter(): void {
    const parametersGroup = this.promptForm.get('parameters') as FormGroup;
    const paramKey = `param${this.parameterKeys.length + 1}`;

    // Create a form group for the parameter with name and value fields
    const paramGroup = this.fb.group({
      name: ['', Validators.required],
      value: [''],
    });

    parametersGroup.addControl(paramKey, paramGroup);
    this.parameterKeys.push(paramKey);
  }

  /**
   * Remove a parameter field
   */
  removeParameter(paramName: string): void {
    const parametersGroup = this.promptForm.get('parameters') as FormGroup;
    parametersGroup.removeControl(paramName);
    this.parameterKeys = this.parameterKeys.filter((key) => key !== paramName);
  }

  /**
   * Check if a parameter field is invalid
   */
  isParameterFieldInvalid(paramKey: string, fieldName: string): boolean {
    const parametersGroup = this.promptForm.get('parameters') as FormGroup;
    const paramGroup = parametersGroup.get(paramKey) as FormGroup;
    if (!paramGroup) return false;

    const control = paramGroup.get(fieldName);
    return !!control && control.invalid && (control.dirty || control.touched);
  }

  /**
   * Navigate back to the prompts list or prompt detail
   */
  cancel(): void {
    this.router.navigate(['app/prompts']);
  }
}
