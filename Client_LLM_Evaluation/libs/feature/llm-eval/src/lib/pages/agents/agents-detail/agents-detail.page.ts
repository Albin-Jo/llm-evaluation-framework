/* Path: libs/feature/llm-eval/src/lib/pages/agents/agent-detail/agent-detail.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { Agent } from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import { AlertService } from '@ngtx-apps/utils/services';
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
    private alertService: AlertService,
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

  loadAgent(id: string): void {
    this.isLoading = true;
    this.error = null;

    this.agentService.getAgent(id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (agent) => {
          this.agent = agent;
          this.isLoading = false;
        },
        error: (error) => {
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

  onEditClick(): void {
    if (this.agentId) {
      this.router.navigate(['app/agents', this.agentId, 'edit']);
    }
  }

  onDeleteClick(): void {
    if (this.agentId && confirm('Are you sure you want to delete this agent? This action cannot be undone.')) {
      this.agentService.deleteAgent(this.agentId)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: () => {
            this.alertService.showAlert({
              show: true,
              message: 'Agent deleted successfully',
              title: 'Success'
            });
            this.router.navigate(['app/agents']);
          },
          error: (error) => {
            this.alertService.showAlert({
              show: true,
              message: 'Failed to delete agent. Please try again.',
              title: 'Error'
            });
            console.error('Error deleting agent:', error);
          }
        });
    }
  }

  onBackClick(): void {
    this.router.navigate(['app/agents']);
  }

  onTestClick(): void {
    if (this.agentId) {
      this.router.navigate(['app/agents', this.agentId, 'test']);
    }
  }

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
}
