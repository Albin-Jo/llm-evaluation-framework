/* Path: libs/feature/llm-eval/src/lib/pages/agents/agent-create-edit/agent-create-edit.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import {
  Agent,
  AgentCreate,
  AgentUpdate,
  AgentDomain,
  AgentResponse
} from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { AlertService } from '@ngtx-apps/utils/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent,
  QracTextAreaComponent // Changed from QracTextareaComponent to QracTextAreaComponent
} from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-agent-create-edit',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracTextBoxComponent,
    QracSelectComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './agents-create-edit.page.html',
  styleUrls: ['./agents-create-edit.page.scss']
})
export class AgentCreateEditPage implements OnInit, OnDestroy {
  agentForm: FormGroup;
  isEditMode = false;
  isLoading = false;
  isSubmitting = false;
  error: string | null = null;
  agentId: string | null = null;

  // Domain options for select dropdown
  domainOptions = [
    { value: AgentDomain.GENERAL, label: 'General' },
    { value: AgentDomain.CUSTOMER_SERVICE, label: 'Customer Service' },
    { value: AgentDomain.TECHNICAL, label: 'Technical' },
    { value: AgentDomain.MEDICAL, label: 'Medical' },
    { value: AgentDomain.LEGAL, label: 'Legal' },
    { value: AgentDomain.FINANCE, label: 'Finance' },
    { value: AgentDomain.EDUCATION, label: 'Education' },
    { value: AgentDomain.OTHER, label: 'Other' }
  ];

  // Status options for select dropdown
  statusOptions = [
    { value: 'true', label: 'Active' },
    { value: 'false', label: 'Inactive' }
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private agentService: AgentService,
    private alertService: AlertService,
    private route: ActivatedRoute,
    private router: Router
  ) {
    this.agentForm = this.createForm();
  }

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.agentId = params.get('id');
      if (this.agentId) {
        this.isEditMode = true;
        this.loadAgent(this.agentId);
      }
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Create the form with default values
   */
  createForm(): FormGroup {
    return this.fb.group({
      name: ['', [Validators.required, Validators.maxLength(255)]],
      description: [''],
      api_endpoint: ['', [Validators.required]],
      domain: [AgentDomain.GENERAL, [Validators.required]],
      is_active: ['true'],
      model_type: [''],
      version: ['1.0.0', [Validators.pattern(/^\d+\.\d+\.\d+$/)]],
      config: ['{}'],
      tags: [''] // Tags will be entered as comma-separated values
    });
  }

  /**
   * Load agent data for editing
   */
  loadAgent(id: string): void {
    this.isLoading = true;
    this.error = null;

    this.agentService.getAgent(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (agent: AgentResponse) => { // Added type
          this.populateForm(agent);
          this.isLoading = false;
        },
        error: (error: Error) => { // Added type
          this.error = 'Failed to load agent details. Please try again.';
          this.alertService.showAlert({
            show: true,
            message: this.error,
            title: 'Error'
          });
          this.isLoading = false;
          console.error('Error loading agent:', error);
        }
      });
  }

  /**
   * Populate form with agent data
   */
  populateForm(agent: Agent): void {
    this.agentForm.patchValue({
      name: agent.name,
      description: agent.description || '',
      api_endpoint: agent.api_endpoint,
      domain: agent.domain,
      is_active: agent.is_active.toString(),
      model_type: agent.model_type || '',
      version: agent.version || '1.0.0',
      config: agent.config ? JSON.stringify(agent.config, null, 2) : '{}',
      tags: agent.tags ? agent.tags.join(', ') : ''
    });
  }

  /**
   * Handle form submission
   */
  onSubmit(): void {
    if (this.agentForm.invalid) {
      // Mark all fields as touched to show validation errors
      this.markFormGroupTouched(this.agentForm);
      return;
    }

    this.isSubmitting = true;
    const formValues = this.agentForm.value;

    // Parse the tags string into an array
    const tags = formValues.tags
      ? formValues.tags.split(',').map((tag: string) => tag.trim()).filter((tag: string) => tag)
      : [];

    // Parse config from string to object
    let config: Record<string, any> = {};
    try {
      config = formValues.config ? JSON.parse(formValues.config) : {};
    } catch (e) {
      this.alertService.showAlert({
        show: true,
        message: 'Invalid JSON in configuration field',
        title: 'Validation Error'
      });
      this.isSubmitting = false;
      return;
    }

    if (this.isEditMode && this.agentId) {
      // Update existing agent
      const updateData: AgentUpdate = {
        name: formValues.name,
        description: formValues.description,
        api_endpoint: formValues.api_endpoint,
        domain: formValues.domain,
        is_active: formValues.is_active === 'true',
        model_type: formValues.model_type,
        version: formValues.version,
        config,
        tags
      };

      this.agentService.updateAgent(this.agentId, updateData)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (agent: AgentResponse) => { // Added type
            this.isSubmitting = false;
            this.alertService.showAlert({
              show: true,
              message: 'Agent updated successfully',
              title: 'Success'
            });
            this.router.navigate(['app/agents', agent.id]);
          },
          error: (error: Error) => { // Added type
            this.isSubmitting = false;
            this.alertService.showAlert({
              show: true,
              message: 'Failed to update agent. Please try again.',
              title: 'Error'
            });
            console.error('Error updating agent:', error);
          }
        });
    } else {
      // Create new agent
      const createData: AgentCreate = {
        name: formValues.name,
        description: formValues.description,
        api_endpoint: formValues.api_endpoint,
        domain: formValues.domain,
        is_active: formValues.is_active === 'true',
        model_type: formValues.model_type,
        version: formValues.version,
        config,
        tags
      };

      this.agentService.createAgent(createData)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (agent: AgentResponse) => { // Added type
            this.isSubmitting = false;
            this.alertService.showAlert({
              show: true,
              message: 'Agent created successfully',
              title: 'Success'
            });
            this.router.navigate(['app/agents', agent.id]);
          },
          error: (error: Error) => { // Added type
            this.isSubmitting = false;
            this.alertService.showAlert({
              show: true,
              message: 'Failed to create agent. Please try again.',
              title: 'Error'
            });
            console.error('Error creating agent:', error);
          }
        });
    }
  }

  /**
   * Recursively mark all form controls as touched
   */
  markFormGroupTouched(formGroup: FormGroup): void {
    Object.keys(formGroup.controls).forEach(key => {
      const control = formGroup.get(key);
      control?.markAsTouched();

      if (control instanceof FormGroup) {
        this.markFormGroupTouched(control);
      }
    });
  }

  /**
   * Return to the previous page
   */
  onCancel(): void {
    if (this.isEditMode && this.agentId) {
      this.router.navigate(['app/agents', this.agentId]);
    } else {
      this.router.navigate(['app/agents']);
    }
  }

  /**
   * Check if a form control has errors and was touched
   */
  hasError(controlName: string, errorName: string): boolean {
    const control = this.agentForm.get(controlName);
    return !!control && control.touched && control.hasError(errorName);
  }
}
