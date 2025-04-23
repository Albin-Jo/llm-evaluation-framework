/* Path: libs/feature/llm-eval/src/lib/pages/prompts/prompts.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';

import { PromptService } from '@ngtx-apps/data-access/services';
import { PromptResponse, PromptFilter } from '@ngtx-apps/data-access/models';
import { AlertService } from '@ngtx-apps/utils/services';
import { QracButtonComponent } from '@ngtx-apps/ui/components';

// Extended filter interface to include additional filter properties
interface ExtendedPromptFilter extends PromptFilter {
  search?: string;
  sort_by?: string;
}

@Component({
  selector: 'app-prompts',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    RouterModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './prompts.page.html',
  styleUrls: ['./prompts.page.scss']
})
export class PromptsPage implements OnInit, OnDestroy {
  // Data properties
  allPrompts: PromptResponse[] = []; // Store all fetched prompts
  prompts: PromptResponse[] = []; // Displayed prompts (paginated subset)
  isLoading = false;
  error: string | null = null;
  noResults = false;

  // Filter properties
  filter: ExtendedPromptFilter = {
    skip: 0,
    limit: 12
  };

  // Pagination properties
  itemsPerPage = 6; // Show 6 items per page
  currentPage = 1;
  totalCount = 0;
  totalPages = 1;
  visiblePages: number[] = [];

  get startItem(): number {
    return ((this.currentPage - 1) * this.itemsPerPage) + 1;
  }

  get endItem(): number {
    return Math.min(this.currentPage * this.itemsPerPage, this.totalCount);
  }

  private destroy$ = new Subject<void>();

  constructor(
    private promptService: PromptService,
    private alertService: AlertService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadPrompts();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadPrompts(): void {
    this.isLoading = true;
    this.noResults = false;
    this.error = null;

    // Remove skip/limit from filter to get all prompts
    const filterWithoutPagination = { ...this.filter };
    delete filterWithoutPagination.skip;
    delete filterWithoutPagination.limit;

    this.promptService.getPrompts(filterWithoutPagination)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: any) => {
          // Store all prompts
          if (response.data && response.total !== undefined) {
            this.allPrompts = response.data;
            this.totalCount = response.total;
          } else {
            this.allPrompts = response;
            this.totalCount = response.length;
          }

          this.noResults = this.allPrompts.length === 0;
          this.isLoading = false;

          // Calculate pagination
          this.totalPages = Math.ceil(this.totalCount / this.itemsPerPage);
          this.updateVisiblePages();

          // Update displayed prompts based on current page
          this.updateDisplayedPrompts();
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
   * Update the displayed prompts based on current page
   */
  updateDisplayedPrompts(): void {
    const startIndex = (this.currentPage - 1) * this.itemsPerPage;
    const endIndex = Math.min(startIndex + this.itemsPerPage, this.allPrompts.length);
    this.prompts = this.allPrompts.slice(startIndex, endIndex);
  }

  /**
   * Update the array of visible page numbers
   */
  updateVisiblePages(): void {
    const maxVisiblePages = 5;
    const pages: number[] = [];

    if (this.totalPages <= maxVisiblePages) {
      // If total pages are less than max visible, show all pages
      for (let i = 1; i <= this.totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Always show first page
      pages.push(1);

      let startPage = Math.max(2, this.currentPage - 1);
      let endPage = Math.min(this.totalPages - 1, this.currentPage + 1);

      // Adjust if we're near the start or end
      if (this.currentPage <= 3) {
        endPage = Math.min(this.totalPages - 1, 4);
      } else if (this.currentPage >= this.totalPages - 2) {
        startPage = Math.max(2, this.totalPages - 3);
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
      if (endPage < this.totalPages - 1) {
        pages.push(-2); // -2 represents ellipsis
      }

      // Always show last page
      if (this.totalPages > 1) {
        pages.push(this.totalPages);
      }
    }

    this.visiblePages = pages;
  }

  /**
   * Navigate to a specific page
   */
  goToPage(page: number): void {
    if (page < 1 || page > this.totalPages) return;

    this.currentPage = page;
    this.updateDisplayedPrompts();
  }

  onFilterChange(filter: ExtendedPromptFilter): void {
    this.filter = { ...this.filter, ...filter };
    this.currentPage = 1;
    this.loadPrompts();
  }

  onCreatePrompt(): void {
    this.router.navigate(['app/prompts/create']);
  }

  onDeletePromptClick(event: Event, promptId: string): void {
    event.stopPropagation(); // Prevent row click
    this.onDeletePrompt(promptId);
  }

  onDeletePrompt(promptId: string): void {
    if (confirm('Are you sure you want to delete this prompt? This action cannot be undone.')) {
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
  }

  onEditPrompt(event: Event, promptId: string): void {
    event.stopPropagation(); // Prevent row click
    this.router.navigate(['app/prompts', promptId, 'edit']);
  }

  onPromptClick(event: Event, promptId: string): void {
    event.stopPropagation(); // Prevent row click
    // Navigate directly to the edit page instead of the view page
    this.router.navigate(['app/prompts', promptId, 'edit']);
  }

  /**
   * Format date to readable string
   */
  formatDate(dateString: string): string {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
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
}
