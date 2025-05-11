/* Path: libs/feature/llm-eval/src/lib/pages/agents/agents.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';

import {
  Agent,
  AgentFilterParams,
  AgentDomain,
  IntegrationType
} from '@ngtx-apps/data-access/models';
import { AgentService } from '@ngtx-apps/data-access/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent
} from '@ngtx-apps/ui/components';
import { AlertService, ConfirmationDialogService, NotificationService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-agents',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracButtonComponent,
    QracTextBoxComponent,
    QracSelectComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './agents.page.html',
  styleUrls: ['./agents.page.scss']
})
export class AgentsPage implements OnInit, OnDestroy {
  agents: Agent[] = [];
  totalCount = 0;
  isLoading = false;
  error: string | null = null;
  currentPage = 1;
  itemsPerPage = 5; // Updated from 10 to match standard
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: AgentFilterParams = {
    page: 1,
    limit: 5, // Updated to match standard
    sortBy: 'created_at',
    sortDirection: 'desc'
  };

  // For filtering by status
  statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: 'true', label: 'Active' },
    { value: 'false', label: 'Inactive' }
  ];

  // Domain options
  domainOptions = [
    { value: '', label: 'All Domains' },
    { value: AgentDomain.GENERAL, label: 'General' },
    { value: AgentDomain.CUSTOMER_SERVICE, label: 'Customer Service' },
    { value: AgentDomain.TECHNICAL, label: 'Technical' },
    { value: AgentDomain.MEDICAL, label: 'Medical' },
    { value: AgentDomain.LEGAL, label: 'Legal' },
    { value: AgentDomain.FINANCE, label: 'Finance' },
    { value: AgentDomain.EDUCATION, label: 'Education' },
    { value: AgentDomain.OTHER, label: 'Other' }
  ];

  // Integration type options
  integrationOptions = [
    { value: '', label: 'All Types' },
    { value: IntegrationType.AZURE_OPENAI, label: 'Azure OpenAI' },
    { value: IntegrationType.MCP, label: 'Model Control Plane (MCP)' },
    { value: IntegrationType.DIRECT_API, label: 'Direct API' },
    { value: IntegrationType.CUSTOM, label: 'Custom' }
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private agentService: AgentService,
    private alertService: AlertService,
    private notificationService: NotificationService,
    private confirmationDialogService: ConfirmationDialogService,
    private router: Router,
    private fb: FormBuilder
  ) {
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      domain: [''],
      integration_type: ['']
    });
  }

  ngOnInit(): void {
    this.setupFilterListeners();
    this.loadAgents();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  setupFilterListeners(): void {
    // Set up search debounce
    this.filterForm.get('search')?.valueChanges
      .pipe(
        debounceTime(400),
        distinctUntilChanged(),
        takeUntil(this.destroy$)
      )
      .subscribe((value: string) => {
        this.filterParams.name = value;
        this.filterParams.page = 1;
        this.loadAgents();
      });

    // Listen to status changes
    this.filterForm.get('status')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        if (value === 'true') {
          this.filterParams.is_active = true;
        } else if (value === 'false') {
          this.filterParams.is_active = false;
        } else {
          this.filterParams.is_active = undefined;
        }
        this.filterParams.page = 1;
        this.loadAgents();
      });

    // Listen to domain changes
    this.filterForm.get('domain')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.domain = value || undefined;
        this.filterParams.page = 1;
        this.loadAgents();
      });

    // Listen to integration type changes
    this.filterForm.get('integration_type')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.integration_type = value as IntegrationType || undefined;
        this.filterParams.page = 1;
        this.loadAgents();
      });
  }

  loadAgents(): void {
    this.isLoading = true;
    this.error = null;

    this.agentService.getAgents(this.filterParams)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.agents = response.agents;
          this.totalCount = response.totalCount;
          this.isLoading = false;
          this.updateVisiblePages();
        },
        error: (error) => {
          this.error = 'Failed to load agents. Please try again.';
          this.notificationService.error(this.error);
          this.isLoading = false;
          console.error('Error loading agents:', error);
        }
      });
  }

  updateVisiblePages(): void {
    const maxVisiblePages = 5;
    const totalPages = Math.ceil(this.totalCount / this.itemsPerPage);
    const pages: number[] = [];

    if (totalPages <= maxVisiblePages) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      pages.push(1);

      let startPage = Math.max(2, this.filterParams.page! - 1);
      let endPage = Math.min(totalPages - 1, this.filterParams.page! + 1);

      if (this.filterParams.page! <= 3) {
        endPage = Math.min(totalPages - 1, 4);
      } else if (this.filterParams.page! >= totalPages - 2) {
        startPage = Math.max(2, totalPages - 3);
      }

      if (startPage > 2) {
        pages.push(-1);
      }

      for (let i = startPage; i <= endPage; i++) {
        pages.push(i);
      }

      if (endPage < totalPages - 1) {
        pages.push(-2);
      }

      if (totalPages > 1) {
        pages.push(totalPages);
      }
    }

    this.visiblePages = pages;
  }

  onPageChange(page: number, event: Event): void {
    event.preventDefault();
    if (page < 1) return;

    this.filterParams.page = page;
    this.loadAgents();
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      domain: '',
      integration_type: ''
    });

    this.filterParams.name = undefined;
    this.filterParams.is_active = undefined;
    this.filterParams.domain = undefined;
    this.filterParams.integration_type = undefined;
    this.filterParams.page = 1;

    this.loadAgents();
  }

  onSortChange(sortBy: string): void {
    if (this.filterParams.sortBy === sortBy) {
      this.filterParams.sortDirection =
        this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.filterParams.sortBy = sortBy as "created_at" | "name" | "domain" | "updated_at";
      this.filterParams.sortDirection = 'desc';
    }

    this.loadAgents();
  }

  onAgentClick(agent: Agent): void {
    this.router.navigate(['app/agents', agent.id]);
  }

  onEditAgent(event: Event, agentId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/agents', agentId, 'edit']);
  }

  onTestAgent(event: Event, agentId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/agents', agentId, 'test']);
  }

  createNewAgent(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/agents/create']);
  }

  confirmDeleteAgent(event: Event, agentId: string): void {
    event.stopPropagation();

    this.confirmationDialogService.confirmDelete('Agent')
      .subscribe(confirmed => {
        if (confirmed) {
          this.deleteAgent(agentId);
        }
      });
  }

  private deleteAgent(agentId: string): void {
    this.agentService.deleteAgent(agentId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.notificationService.success('Agent deleted successfully');
          this.loadAgents();
        },
        error: (error) => {
          this.notificationService.error('Failed to delete agent. Please try again.');
          console.error('Error deleting agent:', error);
        }
      });
  }

  formatDate(dateString: string): string {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      }).format(date);
    } catch (e) {
      return 'Invalid date';
    }
  }

  truncateText(text: string | undefined, maxLength = 100): string {
    if (!text) return '';
    return text.length > maxLength
      ? `${text.substring(0, maxLength)}...`
      : text;
  }

  getStatusLabel(isActive: boolean): string {
    return isActive ? 'Active' : 'Inactive';
  }
}