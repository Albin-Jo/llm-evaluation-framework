/* Path: libs/feature/llm-eval/src/lib/pages/prompts/prompts.page.ts */
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
  QracButtonComponent,
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
    QracButtonComponent,
    QracTextBoxComponent,
    QracSelectComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './prompts.page.html',
  styleUrls: ['./prompts.page.scss']
})
export class PromptsPage implements OnInit, OnDestroy {
  // Data properties
  prompts: PromptResponse[] = []; // Displayed prompts
  isLoading = false;
  error: string | null = null;
  filterForm: FormGroup;
  Math = Math; // For using Math functions in template

  // Filter and pagination properties
  filterParams: PromptFilterParams = {
    page: 1,
    limit: 5,
    sortBy: 'created_at',
    sortDirection: 'desc'
  };

  totalCount = 0;
  visiblePages: number[] = [];

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
        // Update filter params for search
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

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      category: '',
      isTemplate: ''
    });

    // Reset filter params manually
    this.filterParams.name = undefined;
    this.filterParams.category = undefined;
    this.filterParams.isPublic = undefined;
    this.filterParams.page = 1;

    console.log('Filters cleared');
    this.loadPrompts();
  }

  loadPrompts(): void {
    this.isLoading = true;
    this.error = null;

    console.log('Loading prompts with params:', this.filterParams);

    this.promptService.getPrompts(this.filterParams)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: any) => {
          console.log('Prompts response:', response);

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
          console.log(`Loaded ${this.prompts.length} prompts. Total: ${this.totalCount}`);

          // Calculate pagination
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

  /**
   * Update the array of visible page numbers
   */
  updateVisiblePages(): void {
    const maxVisiblePages = 5;
    const itemsPerPage = this.filterParams.limit || 10;
    const totalPages = Math.ceil(this.totalCount / itemsPerPage);
    const pages: number[] = [];

    console.log(`Updating pagination. Total pages: ${totalPages}, Current page: ${this.filterParams.page}`);

    if (totalPages <= maxVisiblePages) {
      // If total pages are less than max visible, show all pages
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Always show first page
      pages.push(1);

      let startPage = Math.max(2, this.filterParams.page! - 1);
      let endPage = Math.min(totalPages - 1, this.filterParams.page! + 1);

      // Adjust if we're near the start or end
      if (this.filterParams.page! <= 3) {
        endPage = Math.min(totalPages - 1, 4);
      } else if (this.filterParams.page! >= totalPages - 2) {
        startPage = Math.max(2, totalPages - 3);
      }

      // Add ellipsis if needed
      if (startPage > 2) {
        pages.push(-1); // -1 represents ellipsis
      }

      // Add middle pages
      for (let i = startPage; i <= endPage; i++) {
        pages.push(i);
      }

      // Add ellipsis if needed
      if (endPage < totalPages - 1) {
        pages.push(-2); // -2 represents ellipsis
      }

      // Always show last page
      if (totalPages > 1) {
        pages.push(totalPages);
      }
    }

    this.visiblePages = pages;
    console.log('Visible pages:', this.visiblePages);
  }

  /**
   * Navigate to a specific page
   */
  onPageChange(page: number, event: Event): void {
    event.preventDefault();
    if (page < 1) return;

    console.log(`Changing to page ${page}`);
    this.filterParams.page = page;
    this.loadPrompts();
  }

  onSortChange(sortBy: string): void {
    console.log(`Sorting by ${sortBy}, current sort: ${this.filterParams.sortBy}, direction: ${this.filterParams.sortDirection}`);

    // Define all valid sort fields
    const validSortFields = ["created_at", "updated_at", "name", "category"];

    // Ensure the sort field is valid
    if (validSortFields.includes(sortBy)) {
      if (this.filterParams.sortBy === sortBy) {
        // Toggle direction if same sort field
        this.filterParams.sortDirection =
          this.filterParams.sortDirection === 'asc' ? 'desc' : 'asc';
        console.log(`Changed sort direction to ${this.filterParams.sortDirection}`);
      } else {
        // Default to desc for new sort field
        this.filterParams.sortBy = sortBy;
        this.filterParams.sortDirection = 'desc';
        console.log(`Changed sort field to ${sortBy} with direction desc`);
      }

      // Reset to page 1 when sorting changes
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
    event.stopPropagation(); // Prevent row click

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
    event.stopPropagation(); // Prevent row click
    this.router.navigate(['app/prompts', promptId, 'edit']);
  }

  onPromptClick(promptId: string): void {
    // Navigate directly to the edit page
    this.router.navigate(['app/prompts', promptId, 'edit']);
  }

  /**
   * Format date to readable string
   */
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

  /**
   * Truncate text to specified length
   */
  truncateText(text: string | undefined, maxLength = 100): string {
    if (!text) return '';
    return text.length > maxLength
      ? `${text.substring(0, maxLength)}...`
      : text;
  }

  /**
   * Get class for template badge
   */
  getTemplateBadgeClass(isTemplate: boolean | undefined): string {
    return isTemplate ? 'template-badge' : 'custom-badge';
  }
}
