import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormGroup, FormBuilder } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';

import { PromptService } from '@ngtx-apps/data-access/services';
import { PromptResponse, PromptFilterParams } from '@ngtx-apps/data-access/models';
import { AlertService, ConfirmationDialogService } from '@ngtx-apps/utils/services';
import {
  
  QracTextBoxComponent,
  QracSelectComponent
} from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-prompts',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    RouterModule,
    
    QracTextBoxComponent,
    QracSelectComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './prompts.page.html',
  styleUrls: ['./prompts.page.scss']
})
export class PromptsPage implements OnInit, OnDestroy {
  prompts: PromptResponse[] = [];
  totalCount = 0;
  isLoading = false;
  error: string | null = null;
  currentPage = 1;
  itemsPerPage = 5; // Updated from 5 to match standard
  Math = Math;
  visiblePages: number[] = [];
  filterForm: FormGroup;

  filterParams: PromptFilterParams = {
    page: 1,
    limit: 5, // Updated from 5 to match standard
    sortBy: 'created_at',
    sortDirection: 'desc'
  };

  // Define category options
  categoryOptions = [
    { value: '', label: 'All Categories' },
    { value: 'RAG', label: 'RAG' },
    { value: 'CHAT', label: 'Chat' },
    { value: 'SUMMARIZATION', label: 'Summarization' },
    { value: 'CLASSIFICATION', label: 'Classification' },
    { value: 'GENERAL', label: 'General' }
  ];

  // Define template status options
  templateOptions = [
    { value: '', label: 'All Visibility' },
    { value: 'true', label: 'Public Only' },
    { value: 'false', label: 'Private Only' }
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private promptService: PromptService,
    private alertService: AlertService,
    private confirmationDialogService: ConfirmationDialogService,
    private router: Router,
    private fb: FormBuilder
  ) {
    this.filterForm = this.fb.group({
      search: [''],
      category: [''],
      isTemplate: ['']
    });
  }

  ngOnInit(): void {
    this.setupFilterListeners();
    this.loadPrompts();
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
        this.filterParams.name = value || undefined;
        this.filterParams.page = 1;
        this.loadPrompts();
      });

    // Listen to category changes
    this.filterForm.get('category')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        this.filterParams.category = value || undefined;
        this.filterParams.page = 1;
        this.loadPrompts();
      });

    // Listen to template status changes
    this.filterForm.get('isTemplate')?.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe((value: string) => {
        if (value === 'true') {
          this.filterParams.isPublic = true;
        } else if (value === 'false') {
          this.filterParams.isPublic = false;
        } else {
          this.filterParams.isPublic = undefined;
        }
        this.filterParams.page = 1;
        this.loadPrompts();
      });
  }

  loadPrompts(): void {
    this.isLoading = true;
    this.error = null;

    this.promptService.getPrompts(this.filterParams)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: any) => {
          // Handle paginated response
          if (response.prompts && response.totalCount !== undefined) {
            this.prompts = response.prompts;
            this.totalCount = response.totalCount;
          } else if (response.items && response.total !== undefined) {
            // Alternative response format
            this.prompts = response.items;
            this.totalCount = response.total;
          } else if (Array.isArray(response)) {
            // Handle array response
            this.prompts = response;
            this.totalCount = response.length;
          } else {
            console.warn('Unexpected response format:', response);
            this.prompts = [];
            this.totalCount = 0;
          }

          this.isLoading = false;
          this.updateVisiblePages();
        },
        error: (err: Error) => {
          this.error = 'Failed to load prompts. Please try again.';
          this.isLoading = false;
          this.alertService.showAlert({
            show: true,
            message: this.error,
            title: 'Error'
          });
          console.error('Error loading prompts:', err);
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
    this.loadPrompts();
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      category: '',
      isTemplate: ''
    });

    this.filterParams.name = undefined;
    this.filterParams.category = undefined;
    this.filterParams.isPublic = undefined;
    this.filterParams.page = 1;

    this.loadPrompts();
  }

  onSortChange(sortBy: string): void {
    const validSortFields = ["created_at", "updated_at", "name", "category"];

    if (validSortFields.includes(sortBy)) {
      if (this.filterParams.sortBy === sortBy) {
        this.filterParams.sortDirection =
          this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        this.filterParams.sortBy = sortBy;
        this.filterParams.sortDirection = 'desc';
      }

      this.filterParams.page = 1;
      this.loadPrompts();
    } else {
      console.warn(`Invalid sort field: ${sortBy}. Using default sort.`);
    }
  }

  onCreatePrompt(): void {
    this.router.navigate(['app/prompts/create']);
  }

  onDeletePromptClick(event: Event, promptId: string): void {
    event.stopPropagation();

    this.confirmationDialogService.confirmDelete('Prompt')
      .subscribe(confirmed => {
        if (confirmed) {
          this.deletePrompt(promptId);
        }
      });
  }

  deletePrompt(promptId: string): void {
    this.promptService.deletePrompt(promptId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.alertService.showAlert({
            show: true,
            message: 'Prompt deleted successfully',
            title: 'Success'
          });
          this.loadPrompts();
        },
        error: (err: Error) => {
          this.alertService.showAlert({
            show: true,
            message: 'Failed to delete prompt. Please try again.',
            title: 'Error'
          });
          console.error('Error deleting prompt:', err);
        }
      });
  }

  onEditPrompt(event: Event, promptId: string): void {
    event.stopPropagation();
    this.router.navigate(['app/prompts', promptId, 'edit']);
  }

  onPromptClick(promptId: string): void {
    // Navigate directly to the edit page
    this.router.navigate(['app/prompts', promptId, 'edit']);
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

  getTemplateBadgeClass(isPublic: boolean | undefined): string {
    return isPublic ? 'public-badge' : 'private-badge';
  }
}
