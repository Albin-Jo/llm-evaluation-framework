/* Path: libs/feature/llm-eval/src/lib/pages/agents/agents-test/agents-test.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { Agent, AgentResponse } from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracTextAreaComponent
} from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-agent-test',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracButtonComponent,
    QracTextBoxComponent,
    QracTextAreaComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './agents-test.page.html',
  styleUrls: ['./agents-test.page.scss']
})
export class AgentTestPage implements OnInit, OnDestroy {
  agent: Agent | null = null;
  testForm: FormGroup;
  isLoading = false;
  isTesting = false;
  error: string | null = null;
  agentId: string | null = null;
  testResult: Record<string, any> | null = null;
  testSuccess = false;
  jsonError: string | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private agentService: AgentService,
    private notificationService: NotificationService,
    private route: ActivatedRoute,
    private router: Router
  ) {
    this.testForm = this.createForm();
  }

  ngOnInit(): void {
    this.route.paramMap
      .pipe(takeUntil(this.destroy$))
      .subscribe(params => {
        this.agentId = params.get('id');
        if (this.agentId) {
          this.loadAgent(this.agentId);
        } else {
          this.error = 'Agent ID not provided';
          this.notificationService.error(this.error);
        }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Create the test form
   */
  createForm(): FormGroup {
    return this.fb.group({
      query: ['', [Validators.required]],
      context: [''],
      parameters: ['{}'] // Additional parameters as JSON
    });
  }

  /**
   * Load agent data
   */
  loadAgent(id: string): void {
    this.isLoading = true;
    this.error = null;

    this.agentService.getAgent(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (agent: AgentResponse) => {
          this.agent = agent;
          this.isLoading = false;
        },
        error: (error) => {
          this.error = 'Failed to load agent details. Please try again.';
          this.notificationService.error(this.error);
          this.isLoading = false;
          console.error('Error loading agent:', error);
        }
      });
  }

  /**
   * Validate JSON parameters
   */
  validateJsonParameters(): boolean {
    const parametersValue = this.testForm.get('parameters')?.value;
    
    if (!parametersValue || parametersValue === '{}') {
      return true;
    }
    
    try {
      JSON.parse(parametersValue);
      this.jsonError = null;
      return true;
    } catch (e) {
      this.jsonError = 'Invalid JSON format in parameters field';
      return false;
    }
  }

  /**
   * Run test with form data
   */
  onSubmit(): void {
    if (this.testForm.invalid || !this.agentId) {
      this.testForm.markAllAsTouched();
      return;
    }
    
    // Validate JSON parameters
    if (!this.validateJsonParameters()) {
      this.notificationService.error(this.jsonError || 'Invalid JSON in parameters field');
      return;
    }

    this.isTesting = true;
    this.testResult = null;
    this.testSuccess = false;
    const formValues = this.testForm.value;

    // Prepare test input
    let testInput: Record<string, any> = {
      query: formValues.query,
    };

    // Add context if provided
    if (formValues.context) {
      testInput['context'] = formValues.context;
    }

    // Parse additional parameters if provided
    try {
      if (formValues.parameters && formValues.parameters !== '{}') {
        const params = JSON.parse(formValues.parameters);
        testInput = { ...testInput, ...params };
      }
    } catch (e) {
      // This should not happen due to validation above
      this.isTesting = false;
      return;
    }

    // Call the test API
    this.agentService.testAgent(this.agentId, testInput)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result: Record<string, any>) => {
          this.testResult = result;
          this.testSuccess = true;
          this.isTesting = false;
          this.notificationService.success('Agent test completed successfully');
        },
        error: (error: any) => {
          this.notificationService.error('Failed to test agent. Please check the response for details.');
          this.isTesting = false;

          // Still show the error response if available
          if (error.error) {
            this.testResult = {
              error: error.error,
              status: error.status,
              message: error.message || 'Test failed'
            };
          }

          console.error('Error testing agent:', error);
        }
      });
  }

  /**
   * Return to agent details
   */
  onBackClick(): void {
    if (this.agentId) {
      this.router.navigate(['app/agents', this.agentId]);
    } else {
      this.router.navigate(['app/agents']);
    }
  }

  /**
   * Format JSON for display
   */
  formatJson(json: Record<string, any>): string {
    try {
      return JSON.stringify(json, null, 2);
    } catch (e) {
      return JSON.stringify({});
    }
  }

  /**
   * Check if a form control has errors
   */
  hasError(controlName: string, errorType?: string): boolean {
    const control = this.testForm.get(controlName);
    if (!control) return false;

    if (errorType) {
      return control.hasError(errorType) && (control.dirty || control.touched);
    }

    return control.invalid && (control.dirty || control.touched);
  }

  /**
   * Get error message for form control
   */
  getErrorMessage(controlName: string): string {
    const control = this.testForm.get(controlName);
    if (!control || !control.errors) return '';

    if (control.errors['required']) {
      return `This field is required`;
    }

    return 'Invalid value';
  }
}