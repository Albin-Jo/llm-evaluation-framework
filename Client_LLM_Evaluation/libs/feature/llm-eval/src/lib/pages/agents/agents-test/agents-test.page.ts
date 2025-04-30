/* Path: libs/feature/llm-eval/src/lib/pages/agents/agent-test/agent-test.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { Agent, AgentResponse } from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { AlertService } from '@ngtx-apps/utils/services';
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
    QracTextBoxComponent
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

  private destroy$ = new Subject<void>();

  constructor(
    private fb: FormBuilder,
    private agentService: AgentService,
    private alertService: AlertService,
    private route: ActivatedRoute,
    private router: Router
  ) {
    this.testForm = this.createForm();
  }

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.agentId = params.get('id');
      if (this.agentId) {
        this.loadAgent(this.agentId);
      } else {
        this.error = 'Agent ID not provided';
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
        error: (error: Error) => {
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
   * Run test with form data
   */
  onSubmit(): void {
    if (this.testForm.invalid || !this.agentId) {
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
      this.alertService.showAlert({
        show: true,
        message: 'Invalid JSON in parameters field',
        title: 'Validation Error'
      });
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
        },
        error: (error: Error) => {
          this.alertService.showAlert({
            show: true,
            message: 'Failed to test agent. Please try again.',
            title: 'Error'
          });
          this.isTesting = false;

          // Still show the error response if available
          if ((error as any).error) {
            this.testResult = {
              error: (error as any).error,
              status: (error as any).status,
              message: (error as any).message || 'Test failed'
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
    return JSON.stringify(json, null, 2);
  }
}
