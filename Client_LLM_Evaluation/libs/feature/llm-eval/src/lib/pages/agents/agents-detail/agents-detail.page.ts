/* Path: libs/feature/llm-eval/src/lib/pages/agents/agents-detail/agents-detail.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, Observable, of, throwError } from 'rxjs';
import { takeUntil, catchError, finalize } from 'rxjs/operators';
import { Agent, AgentResponse, AgentToolsResponse } from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';
import { ConfirmationDialogService } from '@ngtx-apps/utils/services';
import { QracButtonComponent, QracTextBoxComponent, QracSelectComponent } from '@ngtx-apps/ui/components';
import { SimpleJsonViewerComponent } from '../../../components/json-viewer/json-viewer.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

interface ConfigTab {
  label: string;
  field: string;
}

@Component({
  selector: 'app-agent-detail',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracButtonComponent,
    QracTextBoxComponent,
    QracSelectComponent,
    SimpleJsonViewerComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './agents-detail.page.html',
  styleUrls: ['./agents-detail.page.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AgentDetailPage implements OnInit, OnDestroy {
  // Main agent data
  agent: Agent | null = null;
  agentId: string | null = null;

  // Loading and error states
  isLoading = false;
  error: string | null = null;

  // Agent tools
  agentTools: AgentToolsResponse | null = null;
  isLoadingTools = false;
  toolsError: string | null = null;

  // Expandable sections
  expandedSections = {
    basicInfo: true,
    apiConfig: true,
    configuration: false,
    tools: false
  };

  // Configuration tabs
  configTabs: ConfigTab[] = [
    { label: 'Configuration', field: 'config' },
    { label: 'Auth Credentials', field: 'auth_credentials' },
    { label: 'Retry Config', field: 'retry_config' },
    { label: 'Content Filter', field: 'content_filter_config' }
  ];
  selectedConfigTab = 0;

  // Quick test function
  quickTestQuery = '';
  isRunningTest = false;
  quickTestResult: any = null;
  testSuccess = false;

  private destroy$ = new Subject<void>();

  // Caching mechanism
  private static agentCache = new Map<string, Agent>();

  constructor(
    private agentService: AgentService,
    private notificationService: NotificationService,
    private confirmationDialogService: ConfirmationDialogService,
    private route: ActivatedRoute,
    private router: Router,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.route.paramMap.pipe(takeUntil(this.destroy$)).subscribe(params => {
      this.agentId = params.get('id');
      if (this.agentId) {
        // Try to get from cache first
        const cachedAgent = AgentDetailPage.agentCache.get(this.agentId);
        if (cachedAgent) {
          this.agent = cachedAgent;
          this.cdr.markForCheck();

          // Still load fresh data in background
          this.loadAgent(this.agentId, true);
        } else {
          this.loadAgent(this.agentId);
        }
      } else {
        this.error = 'Agent ID not provided';
        this.cdr.markForCheck();
      }
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Copy text to clipboard
   */
  copyToClipboard(text: string): void {
    navigator.clipboard.writeText(text)
      .then(() => {
        this.notificationService.success('Copied to clipboard');
      })
      .catch(err => {
        console.error('Could not copy text: ', err);
        this.notificationService.error('Failed to copy to clipboard');
      });
  }

  /**
   * Load agent details from the API with caching
   */
  loadAgent(id: string, silent: boolean = false): void {
    if (!silent) {
      this.isLoading = true;
      this.error = null;
      this.cdr.markForCheck();
    }

    this.agentService.getAgent(id)
      .pipe(
        takeUntil(this.destroy$),
        catchError(error => {
          if (!silent) {
            this.error = 'Failed to load agent details. Please try again.';
            this.notificationService.error(this.error);
            this.isLoading = false;
            console.error('Error loading agent:', error);
            this.cdr.markForCheck();
          }
          return throwError(() => error);
        }),
        finalize(() => {
          if (!silent) {
            this.isLoading = false;
            this.cdr.markForCheck();
          }
        })
      )
      .subscribe({
        next: (agent: AgentResponse) => {
          this.agent = agent;

          // Update cache
          AgentDetailPage.agentCache.set(id, agent);

          if (!silent) {
            // Load tools if this section is expanded
            if (this.expandedSections.tools) {
              this.loadAgentTools(id);
            }
          }

          this.cdr.markForCheck();
        }
      });
  }

  /**
   * Load agent tools using the API
   */
  loadAgentTools(id: string): void {
    this.isLoadingTools = true;
    this.toolsError = null;
    this.cdr.markForCheck();

    // Check if the API method exists, otherwise fall back to mock data
    if (typeof this.agentService.getAgentTools === 'function') {
      this.agentService.getAgentTools(id)
        .pipe(
          takeUntil(this.destroy$),
          catchError(error => {
            this.toolsError = 'Failed to load agent tools. Please try again.';
            this.isLoadingTools = false;
            this.cdr.markForCheck();
            return this.getMockAgentTools(id);
          }),
          finalize(() => {
            this.isLoadingTools = false;
            this.cdr.markForCheck();
          })
        )
        .subscribe((tools: AgentToolsResponse) => {
          if (tools) {
            this.agentTools = tools;
            this.cdr.markForCheck();
          }
        });
    } else {
      // If the API method doesn't exist, use mock data
      this.getMockAgentTools(id)
        .pipe(
          takeUntil(this.destroy$),
          finalize(() => {
            this.isLoadingTools = false;
            this.cdr.markForCheck();
          })
        )
        .subscribe((tools: AgentToolsResponse) => {
          this.agentTools = tools;
          this.cdr.markForCheck();
        });
    }
  }

  /**
   * Mock implementation for agent tools in case the API method isn't available
   */
  private getMockAgentTools(id: string): Observable<AgentToolsResponse> {
    return of({
      tools: [
        {
          name: 'Sample Tool',
          description: 'This is a sample tool for demonstration',
          parameters: {
            param1: {
              type: 'string',
              description: 'A string parameter'
            },
            param2: {
              type: 'number',
              description: 'A numeric parameter'
            }
          },
          required_parameters: ['param1']
        }
      ]
    });
  }

  /**
   * Run a quick test against the agent
   */
  runQuickTest(): void {
    if (!this.quickTestQuery || !this.agentId) return;

    this.isRunningTest = true;
    this.quickTestResult = null;
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
          this.quickTestResult = result;
          this.testSuccess = true;
          this.cdr.markForCheck();
        },
        error: (error) => {
          this.quickTestResult = {
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
   * Toggle section expansion
   */
  toggleSection(section: string): void {
    if (section in this.expandedSections) {
      this.expandedSections[section as keyof typeof this.expandedSections] =
        !this.expandedSections[section as keyof typeof this.expandedSections];

      // If expanding tools section, load data if needed
      if (this.expandedSections[section as keyof typeof this.expandedSections] && this.agentId) {
        if (section === 'tools' && !this.agentTools) {
          this.loadAgentTools(this.agentId);
        }
      }

      this.cdr.markForCheck();
    }
  }

  /**
   * Select configuration tab
   */
  selectConfigTab(index: number): void {
    if (index >= 0 && index < this.configTabs.length) {
      this.selectedConfigTab = index;
      this.cdr.markForCheck();
    }
  }

  /**
   * Navigate to agent edit page
   */
  onEditClick(): void {
    if (this.agentId) {
      this.router.navigate(['app/agents', this.agentId, 'edit']);
    }
  }

  /**
   * Navigate to agent test page
   */
  onTestClick(): void {
    if (this.agentId) {
      this.router.navigate(['app/agents', this.agentId, 'test']);
    }
  }

  /**
   * Delete agent with confirmation
   */
  onDeleteClick(): void {
    if (!this.agentId || !this.agent) return;

    this.confirmationDialogService.confirmDelete(`agent "${this.agent.name}"`)
      .subscribe(confirmed => {
        if (confirmed) {
          this.deleteAgent(this.agentId!);
        }
      });
  }

  /**
   * Delete agent from the API
   */
  private deleteAgent(id: string): void {
    this.agentService.deleteAgent(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.notificationService.success('Agent deleted successfully');

          // Remove from cache
          AgentDetailPage.agentCache.delete(id);

          this.router.navigate(['app/agents']);
        },
        error: (error) => {
          this.notificationService.error('Failed to delete agent. Please try again.');
          console.error('Error deleting agent:', error);
        }
      });
  }

  /**
   * Navigate back to agents list
   */
  onBackClick(): void {
    this.router.navigate(['app/agents']);
  }

  /**
   * Format date for display
   */
  formatDate(dateString: string | null): string {
    if (!dateString) return 'N/A';

    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    } catch (e) {
      return 'Invalid date';
    }
  }

  /**
   * Format JSON for display
   */
  formatJson(json: any): string {
    if (!json) return '{}';

    try {
      return JSON.stringify(json, null, 2);
    } catch (e) {
      return JSON.stringify({});
    }
  }
}
