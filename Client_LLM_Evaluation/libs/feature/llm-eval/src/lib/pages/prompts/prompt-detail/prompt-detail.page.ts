import { Component, OnDestroy, OnInit, inject, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';

import { PromptService } from '@ngtx-apps/data-access/services';
import { PromptResponse } from '@ngtx-apps/data-access/models';
import { AlertService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-prompt-detail',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './prompt-detail.page.html',
  styleUrls: ['./prompt-detail.page.scss']
})
export class PromptDetailPage implements OnInit, OnDestroy {
  promptId: string | null = null;
  prompt: PromptResponse | null = null;
  isLoading = false;
  error: string | null = null;

  private destroy$ = new Subject<void>();
  private promptService = inject(PromptService);
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private alertService = inject(AlertService);

  ngOnInit(): void {
    this.promptId = this.route.snapshot.paramMap.get('id');
    if (this.promptId) {
      this.loadPrompt();
    } else {
      this.error = 'No prompt ID provided';
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadPrompt(): void {
    if (!this.promptId) return;

    this.isLoading = true;
    this.error = null;

    this.promptService.getPromptById(this.promptId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (prompt: PromptResponse) => {
          this.prompt = prompt;
          this.isLoading = false;
        },
        error: (err: Error) => {
          this.error = 'Failed to load prompt details';
          this.isLoading = false;
          this.alertService.showAlert({
            show: true,
            message: 'Failed to load prompt details. Please try again.',
            title: 'Error'
          });
          console.error('Error loading prompt:', err);
        }
      });
  }

  deletePrompt(): void {
    if (!this.promptId) return;

    if (confirm('Are you sure you want to delete this prompt? This action cannot be undone.')) {
      this.promptService.deletePrompt(this.promptId)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: () => {
            this.alertService.showAlert({
              show: true,
              message: 'Prompt deleted successfully',
              title: 'Success'
            });
            this.router.navigate(['app/prompts']);
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

  goBack(): void {
    this.router.navigate(['app/prompts']);
  }

  editPrompt(): void {
    if (this.promptId) {
      this.router.navigate(['app/prompts', this.promptId, 'edit']);
    }
  }

  hasParameters(parameters: Record<string, any> | undefined): boolean {
    if (!parameters) return false;
    return Object.keys(parameters).length > 0;
  }

  getParameterEntries(parameters: Record<string, any> | undefined): Array<{key: string, value: any}> {
    if (!parameters) return [];
    return Object.entries(parameters).map(([key, value]) => ({ key, value }));
  }

  formatDate(dateString: string): string {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString(undefined, {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }
}
