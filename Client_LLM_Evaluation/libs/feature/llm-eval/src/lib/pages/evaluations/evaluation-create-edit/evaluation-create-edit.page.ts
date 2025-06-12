import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, forkJoin, takeUntil } from 'rxjs';

import {
  EvaluationService,
  AgentService,
  DatasetService,
  PromptService,
} from '@ngtx-apps/data-access/services';
import {
  EvaluationCreate,
  Evaluation,
  EvaluationUpdate,
  EvaluationMethod,
  EvaluationStatus,
  Agent,
  Dataset,
  PromptResponse,
} from '@ngtx-apps/data-access/models';
import { NotificationService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-evaluation-create-edit',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterModule],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './evaluation-create-edit.page.html',
  styleUrls: ['./evaluation-create-edit.page.scss'],
})
export class EvaluationCreateEditPage implements OnInit, OnDestroy {
  selectedTabIndex = 0;

  evaluationForm!: FormGroup;
  isEditMode = false;
  evaluationId: string | null = null;
  isLoading = false;
  isSaving = false;
  error: string | null = null;
  pageTitle = 'Create New Evaluation';

  // Reference data
  agents: Agent[] = [];
  datasets: Dataset[] = [];
  prompts: PromptResponse[] = [];

  // Formatted options for select dropdowns
  agentOptions: { value: string; label: string }[] = [];
  datasetOptions: { value: string; label: string }[] = [];
  promptOptions: { value: string; label: string }[] = [];

  // Available metrics based on dataset type
  availableMetrics: string[] = [];
  selectedMetrics: string[] = [];

  // Method options for select dropdown
  methodOptions = [
    { value: EvaluationMethod.RAGAS, label: 'RAGAS' },
    { value: EvaluationMethod.DEEPEVAL, label: 'DeepEval' },
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private evaluationService: EvaluationService,
    private agentService: AgentService,
    private datasetService: DatasetService,
    private promptService: PromptService,
    private route: ActivatedRoute,
    private router: Router,
    private notificationService: NotificationService
  ) {}

  ngOnInit(): void {
    this.initializeForm();
    this.loadReferenceData();

    this.evaluationId = this.route.snapshot.paramMap.get('id');

    if (this.evaluationId) {
      this.isEditMode = true;
      this.pageTitle = 'Edit Evaluation';
      this.loadEvaluationData();
    }

    // Set up form subscriptions to watch for changes
    this.setupFormSubscriptions();

    // Listen for dataset changes to update available metrics
    this.evaluationForm
      .get('dataset_id')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe((datasetId) => {
        if (datasetId) {
          this.loadDatasetMetrics(datasetId);
        }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Initialize the form with default values and validators
   */
  private initializeForm(): void {
    this.evaluationForm = this.fb.group({
      name: ['', [Validators.required, Validators.maxLength(255)]],
      description: [''],
      method: [EvaluationMethod.RAGAS, Validators.required],
      agent_id: ['', Validators.required],
      dataset_id: ['', Validators.required],
      prompt_id: ['', Validators.required],
      config: this.fb.group({
        temperature: [0.7],
        max_tokens: [1000],
        include_references: [true],
        include_context: [true],
      }),
      metrics: [[]],
    });
  }

  /**
   * Determines if the Next button should be disabled
   * @returns True if the button should be disabled, false otherwise
   */
  isNextButtonDisabled(): boolean {
    if (this.selectedTabIndex === 0) {
      // Check each required field on the Basic Setup tab
      const nameInvalid = this.evaluationForm.get('name')?.invalid === true;
      const methodInvalid = this.evaluationForm.get('method')?.invalid === true;
      const agentInvalid =
        this.evaluationForm.get('agent_id')?.invalid === true;
      const datasetInvalid =
        this.evaluationForm.get('dataset_id')?.invalid === true;
      const promptInvalid =
        this.evaluationForm.get('prompt_id')?.invalid === true;

      return (
        nameInvalid ||
        methodInvalid ||
        agentInvalid ||
        datasetInvalid ||
        promptInvalid
      );
    }

    return false; // Enable the button on other tabs
  }

  /**
   * Determines if the Save/Update button should be disabled
   * @returns True if the button should be disabled, false otherwise
   */
  isSaveButtonDisabled(): boolean {
    // Check form validity
    const formInvalid = this.evaluationForm.invalid === true;

    // Check if metrics are selected
    const noMetricsSelected = this.selectedMetrics.length === 0;

    // Check if saving is in progress
    const isSavingInProgress = this.isSaving === true;

    // Check if the evaluation is editable
    const isNotEditable = !this.isEvaluationEditable;

    return (
      formInvalid || noMetricsSelected || isSavingInProgress || isNotEditable
    );
  }

  /**
   * Load reference data for dropdowns (agents, datasets, prompts)
   */
  private loadReferenceData(): void {
    this.isLoading = true;
    this.error = null;
    forkJoin({
      agents: this.agentService.getAgents(),
      datasets: this.datasetService.getDatasets(),
      prompts: this.promptService.getPrompts(),
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (data) => {
          // Store original data
          this.agents = data.agents.agents || [];
          this.datasets = data.datasets.datasets || [];

          // Handle prompts data with proper type checking for PromptsResponse
          if (
            data.prompts &&
            'prompts' in data.prompts &&
            Array.isArray(data.prompts.prompts)
          ) {
            // New response format with { prompts: [...], totalCount: number }
            this.prompts = data.prompts.prompts;
          } else if (
            data.prompts &&
            'items' in data.prompts &&
            Array.isArray(data.prompts.items)
          ) {
            // Alternative format with { items: [...], total: number }
            this.prompts = data.prompts.items;
          } else if (Array.isArray(data.prompts)) {
            // Direct array format (backwards compatibility)
            this.prompts = data.prompts;
          } else {
            // Fallback to empty array if no valid format is found
            console.warn('Unexpected prompts response format:', data.prompts);
            this.prompts = [];
          }

          // Convert to option format for qrac-select components
          this.agentOptions = this.agents.map((agent) => ({
            value: agent.id,
            label: agent.name,
          }));

          this.datasetOptions = this.datasets.map((dataset) => ({
            value: dataset.id,
            label: dataset.name,
          }));

          this.promptOptions = this.prompts.map((prompt) => ({
            value: prompt.id,
            label: prompt.name,
          }));

          this.isLoading = false;
        },
        error: (err) => {
          this.error = 'Failed to load reference data. Please try again.';
          this.isLoading = false;
          console.error('Error loading reference data:', err);
          this.notificationService.error(this.error);
        },
      });
  }

  /**
   * Set up form subscriptions to watch for dataset and method changes
   */
  private setupFormSubscriptions(): void {
    // Listen for dataset changes to update available metrics
    this.evaluationForm
      .get('dataset_id')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe((datasetId) => {
        if (datasetId) {
          this.loadDatasetMetrics(datasetId);
        } else {
          // Clear metrics when no dataset is selected
          this.availableMetrics = [];
          this.selectedMetrics = [];
          this.evaluationForm.get('metrics')?.setValue([]);
        }
      });

    // Listen for evaluation method changes to update available metrics
    this.evaluationForm
      .get('method')
      ?.valueChanges.pipe(takeUntil(this.destroy$))
      .subscribe((method) => {
        if (method) {
          this.loadDatasetMetricsForMethod();
        } else {
          // Clear metrics when no method is selected
          this.availableMetrics = [];
          this.selectedMetrics = [];
          this.evaluationForm.get('metrics')?.setValue([]);
        }
      });
  }
  /**
   * Load dataset metrics when evaluation method changes
   */
  private loadDatasetMetricsForMethod(): void {
    const datasetId = this.evaluationForm.get('dataset_id')?.value;
    if (datasetId) {
      this.loadDatasetMetrics(datasetId);
    }
  }
  /**
   * Load dataset metrics when a dataset is selected
   */
  private loadDatasetMetrics(datasetId: string): void {
    // Find the selected dataset to get its type
    const selectedDataset = this.datasets.find((d) => d.id === datasetId);
    if (!selectedDataset || !selectedDataset.type) {
      this.availableMetrics = [];
      this.selectedMetrics = [];
      return;
    }

    // Get the selected evaluation method from the form
    const evaluationMethod = this.evaluationForm.get('method')?.value;

    // If no evaluation method is selected yet, default to RAGAS or wait
    if (!evaluationMethod) {
      this.availableMetrics = [];
      this.selectedMetrics = [];
      return;
    }

    this.evaluationService
      .getSupportedMetrics(selectedDataset.type, evaluationMethod)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (metrics) => {
          this.availableMetrics = metrics['supported_metrics'] || [];
          // Reset the selected metrics when changing datasets or methods
          this.selectedMetrics = [];
          this.evaluationForm.get('metrics')?.setValue([]);
        },
        error: (err) => {
          console.error('Error loading metrics for dataset type:', err);
          this.notificationService.error(
            `Failed to load ${evaluationMethod} metrics for the selected dataset type.`
          );
          // Reset metrics on error
          this.availableMetrics = [];
          this.selectedMetrics = [];
          this.evaluationForm.get('metrics')?.setValue([]);
        },
      });
  }
  /**
   * Load evaluation data for edit mode
   */
  loadEvaluationData(): void {
    if (!this.evaluationId) return;

    this.isLoading = true;
    this.error = null;

    this.evaluationService
      .getEvaluation(this.evaluationId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (evaluation) => {
          // Enhanced check for edit permissions
          if (
            this.isEditMode &&
            evaluation.status !== EvaluationStatus.PENDING
          ) {
            this.isLoading = false;

            // Show user-friendly error message based on status
            let errorMessage = '';
            switch (evaluation.status) {
              case EvaluationStatus.RUNNING:
                errorMessage = "Cannot edit evaluation while it's running";
                break;
              case EvaluationStatus.COMPLETED:
                errorMessage = 'Cannot edit completed evaluation';
                break;
              case EvaluationStatus.FAILED:
                errorMessage = 'Cannot edit failed evaluation';
                break;
              case EvaluationStatus.CANCELLED:
                errorMessage = 'Cannot edit cancelled evaluation';
                break;
              default:
                errorMessage = `Cannot edit ${String(
                  evaluation.status
                ).toLowerCase()} evaluation`;
            }

            this.notificationService.error(errorMessage);
            this.router.navigate(['app/evaluations']);
            return;
          }

          this.populateForm(evaluation);
          this.isLoading = false;

          // Load metrics for this dataset AND method combination
          if (evaluation.dataset_id && evaluation.method) {
            this.evaluationForm.patchValue({ method: evaluation.method });
            this.loadDatasetMetrics(evaluation.dataset_id);
          }
        },
        error: (err) => {
          this.error = 'Failed to load evaluation data';
          this.isLoading = false;
          console.error('Error loading evaluation data:', err);
          this.notificationService.error(this.error);
        },
      });
  }

  /**
   * Populate form with existing evaluation data
   */
  populateForm(evaluation: Evaluation): void {
    // Build config form group if needed
    if (evaluation.config) {
      const configGroup = this.fb.group({});

      // Add all existing config properties
      Object.entries(evaluation.config).forEach(([key, value]) => {
        configGroup.addControl(key, this.fb.control(value));
      });

      // Add default properties if not present
      if (!evaluation.config['temperature']) {
        configGroup.addControl('temperature', this.fb.control(0.7));
      }
      if (!evaluation.config['max_tokens']) {
        configGroup.addControl('max_tokens', this.fb.control(1000));
      }
      if (evaluation.config['include_references'] === undefined) {
        configGroup.addControl('include_references', this.fb.control(true));
      }
      if (evaluation.config['include_context'] === undefined) {
        configGroup.addControl('include_context', this.fb.control(true));
      }

      // Replace the config group in the form
      this.evaluationForm.setControl('config', configGroup);
    }

    // Set the selected metrics
    this.selectedMetrics = evaluation.metrics || [];

    // Update form values
    this.evaluationForm.patchValue({
      name: evaluation.name,
      description: evaluation.description || '',
      method: evaluation.method,
      agent_id: evaluation.agent_id,
      dataset_id: evaluation.dataset_id,
      prompt_id: evaluation.prompt_id,
      metrics: evaluation.metrics || [],
    });
  }

  /**
   * Save the evaluation (create or update)
   */
  saveEvaluation(): void {
    if (this.evaluationForm.invalid) {
      this.markFormGroupTouched(this.evaluationForm);
      return;
    }

    this.isSaving = true;
    const formValue = this.evaluationForm.getRawValue();

    if (this.isEditMode) {
      this.updateEvaluation(formValue);
    } else {
      this.createEvaluation(formValue);
    }
  }

  /**
   * Create a new evaluation
   */
  createEvaluation(formData: EvaluationCreate): void {
    this.evaluationService
      .createEvaluation(formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.isSaving = false;
          this.notificationService.success('Evaluation created successfully');
          this.router.navigate(['app/evaluations']);
        },
        error: (err) => {
          this.isSaving = false;
          this.notificationService.error(
            'Failed to create evaluation. Please try again.'
          );
          console.error('Error creating evaluation:', err);
        },
      });
  }

  /**
   * Update an existing evaluation
   */
  updateEvaluation(formData: EvaluationUpdate): void {
    if (!this.evaluationId) return;

    this.evaluationService
      .updateEvaluation(this.evaluationId, formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.isSaving = false;
          this.notificationService.success('Evaluation updated successfully');
          this.router.navigate(['app/evaluations']);
        },
        error: (err) => {
          this.isSaving = false;
          this.notificationService.error(
            'Failed to update evaluation. Please try again.'
          );
          console.error('Error updating evaluation:', err);
        },
      });
  }

  /**
   * Handle metric selection
   */
  toggleMetric(metric: string): void {
    const index = this.selectedMetrics.indexOf(metric);

    if (index >= 0) {
      // Remove the metric
      this.selectedMetrics.splice(index, 1);
    } else {
      // Add the metric
      this.selectedMetrics.push(metric);
    }

    // Update the form control
    this.evaluationForm.get('metrics')?.setValue([...this.selectedMetrics]);
  }

  /**
   * Check if a metric is selected
   */
  isMetricSelected(metric: string): boolean {
    return this.selectedMetrics.includes(metric);
  }

  /**
   * Check if a field is invalid and has been touched
   */
  isFieldInvalid(fieldName: string): boolean {
    const control = this.evaluationForm.get(fieldName);
    return !!control && control.invalid && (control.dirty || control.touched);
  }

  /**
   * Get error message for field validation
   */
  getErrorMessage(fieldName: string): string {
    const control = this.evaluationForm.get(fieldName);
    if (!control || !control.errors) return '';

    if (control.errors['required']) {
      return `${
        fieldName.charAt(0).toUpperCase() + fieldName.slice(1)
      } is required`;
    }
    if (control.errors['maxlength']) {
      return `${
        fieldName.charAt(0).toUpperCase() + fieldName.slice(1)
      } cannot exceed ${control.errors['maxlength'].requiredLength} characters`;
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
   * Navigate back to the evaluations list
   */
  cancel(): void {
    this.router.navigate(['app/evaluations']);
  }

  /**
   * Get flat list of available metrics for UI display
   */
  get flatMetricsList(): string[] {
    const result: string[] = [];
    Object.values(this.availableMetrics).forEach((metrics) => {
      if (Array.isArray(metrics)) {
        result.push(...metrics);
      } else if (typeof metrics === 'string') {
        result.push(metrics);
      }
    });
    return result;
  }

  /**
   * Check if the currently selected evaluation is editable
   */
  get isEvaluationEditable(): boolean {
    // In create mode, always editable
    if (!this.isEditMode) return true;

    // In edit mode, only pending evaluations are editable
    const evaluationStatus = this.evaluationForm.get('status')?.value;
    return evaluationStatus === EvaluationStatus.PENDING;
  }
}
