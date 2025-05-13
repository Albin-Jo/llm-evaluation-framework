/* Path: libs/feature/llm-eval/src/lib/pages/agents/agents-test/agents-test.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, takeUntil, finalize } from 'rxjs';
import { Agent, AgentResponse } from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracTextAreaComponent
} from '@ngtx-apps/ui/components';

interface TestHistoryItem {
  query: string;
  result: Record<string, any>;
  timestamp: Date;
  success: boolean;
}

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
  styleUrls: ['./agents-test.page.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
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
  
  // Track test history to allow users to see previous tests
  testHistory: TestHistoryItem[] = [];
  
  // Show/hide history panel
  showHistory = false;

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private agentService: AgentService,
    private notificationService: NotificationService,
    private route: ActivatedRoute,
    private router: Router,
    private cdr: ChangeDetectorRef
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
          // Try to load test history from session storage
          this.loadTestHistory();
        } else {
          this.error = 'Agent ID not provided';
          this.notificationService.error(this.error);
          this.cdr.markForCheck();
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
    this.cdr.markForCheck();

    this.agentService.getAgent(id)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isLoading = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe({
        next: (agent: AgentResponse) => {
          this.agent = agent;
          // Pre-populate the form with example query if available
          if (agent.domain) {
            this.setDomainSpecificExample(agent.domain);
          }
          this.cdr.markForCheck();
        },
        error: (error) => {
          this.error = 'Failed to load agent details. Please try again.';
          this.notificationService.error(this.error);
          console.error('Error loading agent:', error);
          this.cdr.markForCheck();
        }
      });
  }

  /**
   * Set domain-specific examples in the form
   */
  setDomainSpecificExample(domain: string): void {
    let exampleQuery = '';
    let exampleContext = '';
    
    switch (domain.toLowerCase()) {
      case 'customer_service':
        exampleQuery = 'How do I reset my password?';
        exampleContext = 'User is trying to access their account but has forgotten their password.';
        break;
      case 'technical':
        exampleQuery = 'What are the system requirements for the latest release?';
        exampleContext = 'User is planning to upgrade to the newest software version.';
        break;
      case 'medical':
        exampleQuery = 'What are the common side effects of this medication?';
        exampleContext = 'Patient is asking about a prescription medication.';
        break;
      case 'legal':
        exampleQuery = 'What documents do I need for trademark registration?';
        exampleContext = 'Client is interested in registering a trademark for their new product.';
        break;
      case 'finance':
        exampleQuery = 'What are the current interest rates for savings accounts?';
        exampleContext = 'Customer is looking to open a new savings account.';
        break;
      case 'education':
        exampleQuery = 'What prerequisites are required for this course?';
        exampleContext = 'Student is planning their course schedule for next semester.';
        break;
      default:
        exampleQuery = 'How can I help you today?';
        exampleContext = 'User has initiated a conversation with the agent.';
    }
    
    this.testForm.patchValue({
      query: exampleQuery,
      context: exampleContext
    });
    this.cdr.markForCheck();
  }

  /**
   * Validate JSON parameters
   */
  validateJsonParameters(): boolean {
    const parametersValue = this.testForm.get('parameters')?.value;
    
    if (!parametersValue || parametersValue === '{}') {
      this.jsonError = null;
      return true;
    }
    
    try {
      JSON.parse(parametersValue);
      this.jsonError = null;
      return true;
    } catch (e) {
      this.jsonError = 'Invalid JSON format in parameters field';
      this.cdr.markForCheck();
      return false;
    }
  }

  /**
   * Format JSON input to make it more readable
   */
  formatJsonInput(): void {
    const parametersValue = this.testForm.get('parameters')?.value;
    
    if (!parametersValue || parametersValue === '{}') {
      return;
    }
    
    try {
      const parsed = JSON.parse(parametersValue);
      const formatted = JSON.stringify(parsed, null, 2);
      this.testForm.patchValue({ parameters: formatted });
      this.jsonError = null;
      this.cdr.markForCheck();
    } catch (e) {
      this.jsonError = 'Invalid JSON format in parameters field';
      this.cdr.markForCheck();
    }
  }

  /**
   * Run test with form data
   */
  onSubmit(): void {
    if (this.testForm.invalid || !this.agentId) {
      this.testForm.markAllAsTouched();
      this.cdr.markForCheck();
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
    this.cdr.markForCheck();
    
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
      this.cdr.markForCheck();
      return;
    }

    // Call the test API
    this.agentService.testAgent(this.agentId, testInput)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.isTesting = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe({
        next: (result: Record<string, any>) => {
          this.testResult = result;
          this.testSuccess = true;
          this.notificationService.success('Agent test completed successfully');
          
          // Add to test history
          this.addToTestHistory({
            query: formValues.query,
            result: result,
            timestamp: new Date(),
            success: true
          });
          
          this.cdr.markForCheck();
        },
        error: (error: any) => {
          this.notificationService.error('Failed to test agent. Please check the response for details.');

          // Still show the error response if available
          if (error.error) {
            this.testResult = {
              error: true,
              status: error.status,
              message: error.message || 'Test failed',
              details: error.error
            };
            
            // Add to test history
            this.addToTestHistory({
              query: formValues.query,
              result: this.testResult,
              timestamp: new Date(),
              success: false
            });
          }

          console.error('Error testing agent:', error);
          this.cdr.markForCheck();
        }
      });
  }

  /**
   * Add a test to the history and save to session storage
   */
  private addToTestHistory(test: TestHistoryItem): void {
    this.testHistory.unshift(test);
    
    // Limit history to 10 items
    if (this.testHistory.length > 10) {
      this.testHistory = this.testHistory.slice(0, 10);
    }
    
    // Save to session storage
    try {
      sessionStorage.setItem(`agent_test_history_${this.agentId}`, JSON.stringify(this.testHistory));
    } catch (e) {
      console.warn('Failed to save test history to session storage:', e);
    }
  }
  
  /**
   * Load test history from session storage
   */
  private loadTestHistory(): void {
    try {
      const saved = sessionStorage.getItem(`agent_test_history_${this.agentId}`);
      if (saved) {
        this.testHistory = JSON.parse(saved);
        
        // Convert string timestamps back to Date objects
        this.testHistory.forEach(item => {
          item.timestamp = new Date(item.timestamp);
        });
        
        this.cdr.markForCheck();
      }
    } catch (e) {
      console.warn('Failed to load test history from session storage:', e);
    }
  }
  
  /**
   * Load a previous test from history
   */
  loadFromHistory(test: TestHistoryItem): void {
    this.testForm.patchValue({
      query: test.query,
      // Keep current context and parameters
    });
    
    this.testResult = test.result;
    this.testSuccess = test.success;
    this.cdr.markForCheck();
  }

  /**
   * Clear test history
   */
  clearTestHistory(): void {
    this.testHistory = [];
    sessionStorage.removeItem(`agent_test_history_${this.agentId}`);
    this.cdr.markForCheck();
  }
  
  /**
   * Toggle history panel visibility
   */
  toggleHistory(): void {
    this.showHistory = !this.showHistory;
    this.cdr.markForCheck();
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
  
  /**
   * Get a truncated query string for the history list
   */
  truncateQuery(query: string, maxLength = 30): string {
    if (!query) return '';
    return query.length > maxLength
      ? `${query.substring(0, maxLength)}...`
      : query;
  }
  
  /**
   * Format timestamp for display
   */
  formatTimestamp(date: Date): string {
    if (!date) return '';
    return date.toLocaleTimeString() + ' ' + date.toLocaleDateString();
  }

// Add these methods to the AgentTestPage class

/**
 * Run a quick test with minimal inputs
 */
runQuickTest(): void {
  if (!this.quickTestQuery || !this.agentId) return;

  this.isRunningTest = true;
  this.testResult = null;
  this.testSuccess = false;
  this.cdr.markForCheck();

  const testInput = {
    query: this.quickTestQuery
  };

  this.agentService.testAgent(this.agentId, testInput)
    .pipe(
      takeUntil(this.destroy$),
      finalize(() => {
        this.isRunningTest = false;
        this.cdr.markForCheck();
      })
    )
    .subscribe({
      next: (result) => {
        this.testResult = result;
        this.testSuccess = true;
        this.cdr.markForCheck();
      },
      error: (error) => {
        this.testResult = {
          error: true,
          message: error.message || 'Test failed',
          details: error.error
        };
        this.testSuccess = false;
        this.cdr.markForCheck();
      }
    });
}

/**
 * Extract response text from various possible response formats
 */
getResponseText(): string {
  if (!this.testResult) return '';

  // Handle different response formats
  if (this.testResult.answer) return this.testResult.answer;
  if (this.testResult.response) return this.testResult.response;
  if (this.testResult.message) return this.testResult.message;
  if (this.testResult.text) return this.testResult.text;
  if (this.testResult.content) return this.testResult.content;

  // If the result is a direct string
  if (typeof this.testResult === 'string') return this.testResult;

  // Try to stringify the result if it's an object without known properties
  if (typeof this.testResult === 'object') {
    try {
      // Exclude error details and metadata for cleaner display
      const { error, status, details, processing_time_ms, time, tokens, ...contentProps } = this.testResult;

      // If we still have properties to show
      if (Object.keys(contentProps).length > 0) {
        return JSON.stringify(contentProps, null, 2);
      }
    } catch (e) {}
  }

  return 'No response content available';
}
}