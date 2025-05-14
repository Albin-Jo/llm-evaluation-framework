import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, takeUntil, finalize } from 'rxjs';
import { Agent, AgentResponse } from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';
import { QracButtonComponent } from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-agent-test',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracButtonComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './agents-test.page.html',
  styleUrls: ['./agents-test.page.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AgentTestPage implements OnInit, OnDestroy {
  agent: Agent | null = null;
  isLoading = false;
  error: string | null = null;
  agentId: string | null = null;
  testResult: Record<string, any> | null = null;
  testSuccess = false;
  
  // Chat interface properties
  testQuery = '';
  isRunningTest = false;

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
    this.route.paramMap
      .pipe(takeUntil(this.destroy$))
      .subscribe(params => {
        this.agentId = params.get('id');
        if (this.agentId) {
          this.loadAgent(this.agentId);
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
    
    switch (domain.toLowerCase()) {
      case 'customer_service':
        exampleQuery = 'How do I reset my password?';
        break;
      case 'technical':
        exampleQuery = 'What are the system requirements for the latest release?';
        break;
      case 'medical':
        exampleQuery = 'What are the common side effects of this medication?';
        break;
      case 'legal':
        exampleQuery = 'What documents do I need for trademark registration?';
        break;
      case 'finance':
        exampleQuery = 'What are the current interest rates for savings accounts?';
        break;
      case 'education':
        exampleQuery = 'What prerequisites are required for this course?';
        break;
      default:
        exampleQuery = 'Are you a bot?';
    }
    
    this.testQuery = exampleQuery;
    this.cdr.markForCheck();
  }

  /**
   * Run a test with query
   */
  runTest(): void {
    if (!this.testQuery || !this.agentId) return;

    this.isRunningTest = true;
    this.testResult = null;
    this.testSuccess = false;
    this.cdr.markForCheck();
    
    const testInput = {
      query: this.testQuery
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
          this.notificationService.error('Failed to test agent. Please check the response for details.');

          // Still show the error response if available
          if (error.error) {
            this.testResult = {
              error: true,
              status: error.status,
              message: error.message || 'Test failed',
              details: error.error
            };
          }

          console.error('Error testing agent:', error);
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
    if (this.testResult['answer']) return this.testResult['answer'];
    if (this.testResult['response']) return this.testResult['response'];
    if (this.testResult['message']) return this.testResult['message'];
    if (this.testResult['text']) return this.testResult['text'];
    if (this.testResult['content']) return this.testResult['content'];
    
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
}