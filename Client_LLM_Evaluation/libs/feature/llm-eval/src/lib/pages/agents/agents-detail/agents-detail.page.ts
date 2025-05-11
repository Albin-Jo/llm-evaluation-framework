/* Path: libs/feature/llm-eval/src/lib/pages/agents/agents-detail/agents-detail.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { Agent, AgentResponse } from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { NotificationService } from '@ngtx-apps/utils/services';
import { ConfirmationDialogService } from '@ngtx-apps/utils/services';
import { QracButtonComponent } from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-agent-detail',
  standalone: true,
  imports: [
    CommonModule,
    QracButtonComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './agents-detail.page.html',
  styleUrls: ['./agents-detail.page.scss']
})
export class AgentDetailPage implements OnInit, OnDestroy {
  agent: Agent | null = null;
  isLoading = false;
  error: string | null = null;
  agentId: string | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private agentService: AgentService,
    private notificationService: NotificationService,
    private confirmationDialogService: ConfirmationDialogService,
    private route: ActivatedRoute,
    private router: Router
  ) {}

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
   * Load agent details from the API
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
  formatDate(dateString: string): string {
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
    try {
      return JSON.stringify(json, null, 2);
    } catch (e) {
      return JSON.stringify({});
    }
  }
}