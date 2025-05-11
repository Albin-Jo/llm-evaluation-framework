/* Path: libs/feature/llm-eval/src/lib/pages/agents/agents-detail/agents-detail.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, BehaviorSubject, interval, Observable, of, throwError } from 'rxjs';
import { takeUntil, switchMap, catchError, finalize, tap, startWith, map, share } from 'rxjs/operators';
import { Agent, AgentResponse, AgentHealthResponse, AgentToolsResponse } from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';
import { ConfirmationDialogService } from '@ngtx-apps/utils/services';
import { QracButtonComponent, QracTextBoxComponent, QracSelectComponent } from '@ngtx-apps/ui/components';
import { JsonViewerComponent } from '../../../components/json-viewer/json-viewer.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

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
    JsonViewerComponent
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

  // Health check
  healthStatus: AgentHealthResponse | null = null;
  isCheckingHealth = false;
  healthError: string | null = null;

  // Agent tools
  agentTools: AgentToolsResponse | null = null;
  isLoadingTools = false;
  toolsError: string | null = null;

  // Expandable sections
  expandedSections = {
    basicInfo: true,
    apiConfig: true,
    configuration: false,
    tools: false,
    health: false
  };

  // Quick test function
  quickTestQuery = '';
  isRunningTest = false;
  quickTestResult: any = null;

  // Stats
  usageStats = {
    totalRequests: 0,
    successRate: 0,
    avgLatency: 0,
    lastUsed: null as Date | null
  };

  // Automated health check (every 30 seconds)
  private healthCheckEnabled = false;

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
    this.disableHealthCheck();
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
            // Load tools and health data on initial load
            this.loadAgentTools(id);
            this.checkAgentHealth(id);

            // Load usage stats if available
            this.loadUsageStats(id);
          }

          this.cdr.markForCheck();
        }
      });
  }

  /**
   * Load agent tools - Mock implementation for now
   */
  loadAgentTools(id: string): void {
    this.isLoadingTools = true;
    this.toolsError = null;
    this.cdr.markForCheck();

    // Use our mock implementation since API method is not available yet
    this.getAgentTools(id)
      .pipe(
        takeUntil(this.destroy$),
        catchError(error => {
          this.toolsError = 'Failed to load agent tools. Please try again.';
          this.isLoadingTools = false;
          this.cdr.markForCheck();
          return of(null as unknown as AgentToolsResponse);
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
  }

  /**
   * Check agent health status - Mock implementation for now
   */
  checkAgentHealth(id: string): void {
    this.isCheckingHealth = true;
    this.healthError = null;
    this.cdr.markForCheck();

    // Use our mock implementation since API method is not available yet
    this.mockCheckAgentHealth(id)
      .pipe(
        takeUntil(this.destroy$),
        catchError(error => {
          this.healthError = 'Failed to check agent health. Please try again.';
          this.isCheckingHealth = false;
          this.cdr.markForCheck();
          return of(null as unknown as AgentHealthResponse);
        }),
        finalize(() => {
          this.isCheckingHealth = false;
          this.cdr.markForCheck();
        })
      )
      .subscribe((health: AgentHealthResponse) => {
        if (health) {
          this.healthStatus = health;
          this.cdr.markForCheck();
        }
      });
  }

  /**
   * Enable automated health check (every 30 seconds)
   */
  enableHealthCheck(): void {
    if (this.healthCheckEnabled || !this.agentId) return;

    this.healthCheckEnabled = true;

    // Check initially and then every 30 seconds
    interval(30000)
      .pipe(
        startWith(0),
        takeUntil(this.destroy$),
        switchMap(() => this.mockCheckAgentHealth(this.agentId!))
      )
      .subscribe({
        next: (health: AgentHealthResponse) => {
          this.healthStatus = health;
          this.cdr.markForCheck();
        },
        error: (error) => {
          console.error('Health check error:', error);
        }
      });
  }

  /**
   * Disable automated health check
   */
  disableHealthCheck(): void {
    this.healthCheckEnabled = false;
  }

  /**
   * Load usage statistics for the agent - Simulated
   */
  loadUsageStats(id: string): void {
    // Placeholder for real implementation
    // This would be connected to a real API endpoint

    // Simulate loading stats
    setTimeout(() => {
      this.usageStats = {
        totalRequests: Math.floor(Math.random() * 1000),
        successRate: Math.random() * 100,
        avgLatency: Math.random() * 500,
        lastUsed: new Date(Date.now() - Math.floor(Math.random() * 86400000))
      };
      this.cdr.markForCheck();
    }, 700);
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
          this.cdr.markForCheck();
        },
        error: (error) => {
          this.quickTestResult = {
            error: true,
            message: error.message || 'Test failed',
            details: error.error
          };
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

      // If expanding tools or health sections, load data if needed
      if (this.expandedSections[section as keyof typeof this.expandedSections] && this.agentId) {
        if (section === 'tools' && !this.agentTools) {
          this.loadAgentTools(this.agentId);
        } else if (section === 'health') {
          this.checkAgentHealth(this.agentId);

          // Enable automated health check when health section is expanded
          if (this.expandedSections.health) {
            this.enableHealthCheck();
          } else {
            this.disableHealthCheck();
          }
        }
      }

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
   * Toggle agent active status
   */
  toggleAgentStatus(): void {
    if (!this.agentId || !this.agent) return;

    const newStatus = !this.agent.is_active;
    const actionText = newStatus ? 'activate' : 'deactivate';

    this.confirmationDialogService.confirm({
      title: `${newStatus ? 'Activate' : 'Deactivate'} Agent`,
      message: `Are you sure you want to ${actionText} this agent?`,
      confirmText: 'Yes',
      cancelText: 'No'
    }).subscribe(confirmed => {
      if (confirmed) {
        this.updateAgentStatus(this.agentId!, newStatus);
      }
    });
  }

  /**
   * Update agent status
   */
  private updateAgentStatus(id: string, isActive: boolean): void {
    const updateData = {
      is_active: isActive
    };

    this.agentService.updateAgent(id, updateData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (agent) => {
          this.agent = agent;
          this.notificationService.success(`Agent ${isActive ? 'activated' : 'deactivated'} successfully`);

          // Update cache
          if (AgentDetailPage.agentCache.has(id)) {
            AgentDetailPage.agentCache.set(id, agent);
          }

          this.cdr.markForCheck();
        },
        error: (error) => {
          this.notificationService.error(`Failed to ${isActive ? 'activate' : 'deactivate'} agent. Please try again.`);
          console.error('Error updating agent status:', error);
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

  /**
   * Get health status class
   */
  getHealthStatusClass(): string {
    if (!this.healthStatus) return '';

    return this.healthStatus.healthy ? 'status-healthy' : 'status-unhealthy';
  }

  /**
   * Get percentage format for display
   */
  formatPercentage(value: number): string {
    return value.toFixed(1) + '%';
  }

  /**
   * Format milliseconds for display
   */
  formatMilliseconds(ms: number): string {
    return ms.toFixed(0) + ' ms';
  }

  /**
   * Mock implementation of getAgentTools - Replace when API is available
   */
  private getAgentTools(id: string): Observable<AgentToolsResponse> {
    // Return mock data for now
    return of({
      tools: [
        {
          name: 'Sample Tool',
          description: 'This is a sample tool for demonstration',
          parameters: {
            param1: 'string',
            param2: 'number'
          },
          required_parameters: ['param1']
        }
      ]
    });
  }

  /**
   * Mock implementation of checkAgentHealth - Replace when API is available
   */
  private mockCheckAgentHealth(id: string): Observable<AgentHealthResponse> {
    // Simulate API call
    return of({
      status: 'ok',
      healthy: true,
      message: 'Agent is responding normally',
      details: {
        latency: '120ms',
        lastCheck: new Date().toISOString()
      }
    });
  }
}