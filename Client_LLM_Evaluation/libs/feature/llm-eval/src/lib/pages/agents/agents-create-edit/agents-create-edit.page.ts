/* Path: libs/feature/llm-eval/src/lib/pages/agents/agents-create-edit/agents-create-edit.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, FormsModule, ReactiveFormsModule, AbstractControl } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, finalize, of, takeUntil, BehaviorSubject } from 'rxjs';
import { catchError, debounceTime, distinctUntilChanged } from 'rxjs/operators';
import {
  Agent,
  AgentCreate,
  AgentUpdate,
  AgentDomain,
  AgentResponse,
  IntegrationType,
  AuthType
} from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent,
  QracTextAreaComponent
} from '@ngtx-apps/ui/components';
import { JsonEditorComponent } from '../../../components/json-editor/json-editor.component';

@Component({
  selector: 'app-agent-create-edit',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracButtonComponent,
    QracTextBoxComponent,
    QracSelectComponent,
    QracTextAreaComponent,
    JsonEditorComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './agents-create-edit.page.html',
  styleUrls: ['./agents-create-edit.page.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AgentCreateEditPage implements OnInit, OnDestroy {
  // Form and state management
  agentForm!: FormGroup;
  isEditMode = false;
  isLoading = false;
  isSubmitting = false;
  error: string | null = null;
  agentId: string | null = null;
  
  // Tab state management
  selectedTabIndex = 0;
  tabsVisited = new Set<number>([0]);
  
  // JSON validation tracking
  jsonErrors: { [key: string]: string } = {};
  jsonCache: { [key: string]: any } = {};

  // JSON editor states
  jsonEditorStates: { [key: string]: boolean } = {
    auth_credentials: false,
    request_template: false,
    config: false,
    retry_config: false,
    content_filter_config: false
  };

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

  // Integration type options
  integrationTypeOptions = [
    { value: IntegrationType.AZURE_OPENAI, label: 'Azure OpenAI' },
    { value: IntegrationType.MCP, label: 'Model Control Plane (MCP)' },
    { value: IntegrationType.DIRECT_API, label: 'Direct API' },
    { value: IntegrationType.CUSTOM, label: 'Custom' }
  ];

  // Auth type options
  authTypeOptions = [
    { value: AuthType.API_KEY, label: 'API Key' },
    { value: AuthType.BEARER_TOKEN, label: 'Bearer Token' },
    { value: AuthType.NONE, label: 'None' }
  ];

  // List of JSON field names for validation
  private jsonFields = ['config', 'auth_credentials', 'request_template', 'retry_config', 'content_filter_config'];
  
  // Loading state for individual tabs
  private tabLoadingState = new BehaviorSubject<{[key: number]: boolean}>({
    0: false,
    1: false,
    2: false
  });

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private agentService: AgentService,
    private notificationService: NotificationService,
    private route: ActivatedRoute,
    private router: Router,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    // Initialize the form once
    this.initializeForm();
    
    // Check route params for edit mode
    this.route.paramMap
      .pipe(takeUntil(this.destroy$))
      .subscribe(params => {
        this.agentId = params.get('id');
        if (this.agentId) {
          this.isEditMode = true;
          this.loadAgent(this.agentId);
        }
        this.cdr.markForCheck();
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Initialize form with validation
   */
  private initializeForm(): void {
    this.agentForm = this.fb.group({
      // Basic Information Tab
      name: ['', [Validators.required, Validators.maxLength(255)]],
      description: [''],
      domain: [AgentDomain.GENERAL, [Validators.required]],
      is_active: ['true'],
      model_type: [''],
      version: ['1.0.0', [Validators.pattern(/^\d+\.\d+\.\d+$/)]],
      tags: [''],
      
      // API Configuration Tab
      api_endpoint: ['', [Validators.required]],
      integration_type: [IntegrationType.AZURE_OPENAI],
      auth_type: [AuthType.API_KEY],
      auth_credentials: ['{}'],
      request_template: ['{}'],
      response_format: [''],
      
      // Advanced Configuration Tab
      config: ['{}'],
      retry_config: ['{}'],
      content_filter_config: ['{}']
    });
  }

  /**
   * Update JSON field value from JSON editor
   */
   updateJsonField(fieldName: string, value: string): void {
    try {
      // Validate the JSON
      JSON.parse(value);
      
      // Update form control with the stringified value
      this.agentForm.get(fieldName)?.setValue(value, { emitEvent: true });
      
      // Update cache
      this.jsonCache[fieldName] = JSON.parse(value);
      
      // Clear any errors
      this.jsonErrors[fieldName] = '';
      
      this.cdr.markForCheck();
    } catch (error) {
      this.jsonErrors[fieldName] = `Invalid JSON format in ${this.formatFieldName(fieldName)}`;
      console.warn(`JSON validation error in ${fieldName}:`, error);
      this.cdr.markForCheck();
    }
  }

  /**
 * Format JSON data for the editor
 */
getFormattedJson(fieldName: string): string {
  try {
    const value = this.agentForm.get(fieldName)?.value;
    
    // Check if value is undefined or null
    if (!value || value === 'undefined' || value === 'null') {
      return '{}';
    }
    
    // If it's already an object, stringify it
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2);
    }
    
    // If it's a string, parse and re-stringify it for proper formatting
    if (typeof value === 'string') {
      try {
        const parsed = JSON.parse(value);
        return JSON.stringify(parsed, null, 2);
      } catch {
        // If parsing fails, return as is
        return value;
      }
    }
    
    return '{}';
  } catch (error) {
    console.warn(`Error formatting JSON for ${fieldName}:`, error);
    return '{}';
  }
}

/**
 * Check if a JSON field is valid
 */
 isJsonFieldValid(fieldName: string, isValid: boolean): void {
  if (!isValid) {
    this.jsonErrors[fieldName] = `Invalid JSON format in ${this.formatFieldName(fieldName)}`;
  } else {
    this.jsonErrors[fieldName] = '';
  }
  this.cdr.markForCheck();
}
  
  /**
   * Load agent data for editing with optimized loading and error handling
   */
  loadAgent(id: string): void {
    this.isLoading = true;
    this.error = null;
    this.cdr.markForCheck();

    this.agentService.getAgent(id)
      .pipe(
        takeUntil(this.destroy$),
        catchError(err => {
          this.error = 'Failed to load agent details. Please try again.';
          this.notificationService.error(this.error);
          this.isLoading = false;
          console.error('Error loading agent:', err);
          this.cdr.markForCheck();
          return of(null);
        }),
        finalize(() => {
          this.isLoading = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe((agent: AgentResponse | null) => {
        if (agent) {
          this.populateForm(agent);
        }
      });
  }

  /**
 * Populate form with agent data - optimized for performance
 */
populateForm(agent: Agent): void {
  try {
    // Pre-parse JSON fields to avoid multiple parsing operations
    const parsedData: {[key: string]: any} = {};
    
    // Process JSON fields and cache them
    this.jsonFields.forEach(field => {
      // Only attempt to parse if the field exists on the agent object
      if (agent[field as keyof Agent]) {
        try {
          // For string values that need parsing
          if (typeof agent[field as keyof Agent] === 'string') {
            parsedData[field] = JSON.parse(agent[field as keyof Agent] as string);
          } else {
            // For objects that are already parsed
            parsedData[field] = agent[field as keyof Agent];
          }
          // Cache the parsed data
          this.jsonCache[field] = parsedData[field];
        } catch (e) {
          console.warn(`Failed to parse ${field}:`, e);
          parsedData[field] = {};
          this.jsonCache[field] = {};
        }
      } else {
        // Default to empty object
        parsedData[field] = {};
        this.jsonCache[field] = {};
      }
    });

    // Format JSON fields for display with proper formatting
    const configStr = JSON.stringify(parsedData['config'] || {}, null, 2);
    const authCredentialsStr = JSON.stringify(parsedData['auth_credentials'] || {}, null, 2);
    const requestTemplateStr = JSON.stringify(parsedData['request_template'] || {}, null, 2);
    const retryConfigStr = JSON.stringify(parsedData['retry_config'] || {}, null, 2);
    const contentFilterConfigStr = JSON.stringify(parsedData['content_filter_config'] || {}, null, 2);
    
    // Format tags as comma-separated string (do this only once)
    const tagsStr = agent.tags ? agent.tags.join(', ') : '';

    // Use patchValue with emitEvent: false for better performance
    this.agentForm.patchValue({
      // Basic Information Tab
      name: agent.name,
      description: agent.description || '',
      domain: agent.domain,
      is_active: agent.is_active.toString(),
      model_type: agent.model_type || '',
      version: agent.version || '1.0.0',
      tags: tagsStr,
      
      // API Configuration Tab
      api_endpoint: agent.api_endpoint,
      integration_type: agent.integration_type || IntegrationType.AZURE_OPENAI,
      auth_type: agent.auth_type || AuthType.API_KEY,
      auth_credentials: authCredentialsStr,
      request_template: requestTemplateStr,
      response_format: agent.response_format || '',
      
      // Advanced Configuration Tab
      config: configStr,
      retry_config: retryConfigStr,
      content_filter_config: contentFilterConfigStr
    }, { emitEvent: true }); // Need to trigger valueChanges for JSON editors
    
    // Clear any JSON errors
    this.jsonFields.forEach(field => {
      this.jsonErrors[field] = '';
    });
    
    this.cdr.markForCheck();
  } catch (e) {
    console.error('Error populating form:', e);
    this.notificationService.error('Failed to load agent data correctly');
  }
}

  /**
   * Handle form submission with optimized processing
   */
  onSubmit(): void {
    if (this.agentForm.invalid) {
      // Mark all fields as touched to show validation errors
      this.markFormGroupTouched(this.agentForm);
      return;
    }

    // Validate all JSON fields
    if (this.hasJsonErrors()) {
      this.notificationService.error('Please correct the JSON errors before submitting');
      return;
    }

    this.isSubmitting = true;
    this.cdr.markForCheck();
    
    const formValues = this.agentForm.value;

    // Parse the tags string into an array
    const tags = formValues.tags
      ? formValues.tags.split(',').map((tag: string) => tag.trim()).filter((tag: string) => tag)
      : [];

    // Build agent data using cached JSON objects where possible
    const agentData = {
      name: formValues.name,
      description: formValues.description,
      api_endpoint: formValues.api_endpoint,
      domain: formValues.domain,
      is_active: formValues.is_active === 'true',
      model_type: formValues.model_type,
      version: formValues.version,
      tags,
      integration_type: formValues.integration_type,
      auth_type: formValues.auth_type,
      response_format: formValues.response_format,
      
      // Use cached JSON objects when possible (fallback to parsing)
      config: this.getJsonValue('config'),
      auth_credentials: this.getJsonValue('auth_credentials'),
      request_template: this.getJsonValue('request_template'),
      retry_config: this.getJsonValue('retry_config'),
      content_filter_config: this.getJsonValue('content_filter_config')
    };

    if (this.isEditMode && this.agentId) {
      this.updateAgent(this.agentId, agentData);
    } else {
      this.createAgent(agentData);
    }
  }
  
  /**
   * Get JSON value from cache or parse it
   */
  private getJsonValue(fieldName: string): any {
    // If we have a cached version, use it
    if (this.jsonCache[fieldName]) {
      return this.jsonCache[fieldName];
    }
    
    // Otherwise, try to parse it
    try {
      const value = this.agentForm.get(fieldName)?.value;
      return value && value !== '{}' ? JSON.parse(value) : {};
    } catch (e) {
      console.warn(`Error parsing ${fieldName}:`, e);
      return {};
    }
  }

  /**
   * Create a new agent with optimized request handling
   */
  createAgent(agentData: AgentCreate): void {
    this.agentService.createAgent(agentData)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isSubmitting = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe({
        next: (agent: AgentResponse) => {
          this.notificationService.success('Agent created successfully');
          this.router.navigate(['app/agents', agent.id]);
        },
        error: (error) => {
          this.notificationService.error('Failed to create agent. Please try again.');
          console.error('Error creating agent:', error);
        }
      });
  }

  /**
   * Check if a field is invalid and has been touched
   */
  isFieldInvalid(controlName: string): boolean {
    const control = this.agentForm.get(controlName);
    return !!control && control.invalid && (control.dirty || control.touched);
  }

  /**
   * Get formatted error message for a field
   */
  getErrorMessage(controlName: string): string {
    const control = this.agentForm.get(controlName);
    if (!control || !control.errors) return '';
    
    const errors = Object.keys(control.errors);
    if (errors.length === 0) return '';
    
    const errorType = errors[0];
    
    switch (errorType) {
      case 'required':
        return `${this.formatFieldName(controlName)} is required`;
      case 'maxlength':
        return `${this.formatFieldName(controlName)} exceeds maximum length`;
      case 'pattern':
        return `${this.formatFieldName(controlName)} has invalid format`;
      default:
        return 'Invalid value';
    }
  }

  /**
   * Update an existing agent with optimized request handling
   */
  updateAgent(id: string, agentData: AgentUpdate): void {
    this.agentService.updateAgent(id, agentData)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isSubmitting = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe({
        next: (agent: AgentResponse) => {
          this.notificationService.success('Agent updated successfully');
          this.router.navigate(['app/agents', agent.id]);
        },
        error: (error) => {
          this.notificationService.error('Failed to update agent. Please try again.');
          console.error('Error updating agent:', error);
        }
      });
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
    
    this.cdr.markForCheck();
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
   * Check if the current tab has validation errors
   */
  isTabInvalid(tabIndex: number): boolean {
    switch (tabIndex) {
      case 0: // Basic Information Tab
        return (
          this.isFieldInvalid('name') || 
          this.isFieldInvalid('domain')
        );
      case 1: // API Configuration Tab
        return (
          this.isFieldInvalid('api_endpoint') || 
          !!this.jsonErrors['auth_credentials']
        );
      default:
        return false;
    }
  }

  /**
   * Select a specific tab
   */
  selectTab(index: number): void {
    this.selectedTabIndex = index;
    this.tabsVisited.add(index);
    this.cdr.markForCheck();
  }

  /**
   * Check if there are any JSON validation errors
   */
  hasJsonErrors(): boolean {
    return Object.values(this.jsonErrors).some(error => error !== '');
  }

  /**
   * Format field name for display
   */
  private formatFieldName(name: string): string {
    return name
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }
}