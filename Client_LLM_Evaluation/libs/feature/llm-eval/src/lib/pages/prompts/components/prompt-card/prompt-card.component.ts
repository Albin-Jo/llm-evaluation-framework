import { Component, EventEmitter, Input, Output, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { PromptResponse } from '@ngtx-apps/data-access/models';
import { QracButtonComponent } from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-prompt-card',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './prompt-card.component.html',
  styleUrls: ['./prompt-card.component.scss']
})
export class PromptCardComponent {
  @Input() prompt!: PromptResponse;
  @Output() delete = new EventEmitter<string>();

  /**
   * Format the date to a readable format
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
   * Truncate text to a specified length
   */
  truncateText(text: string, maxLength = 100): string {
    if (!text) return '';
    return text.length > maxLength
      ? `${text.substring(0, maxLength)}...`
      : text;
  }

  /**
   * Handle delete button click
   * @param event The click event
   */
  onDeleteClick(event: Event): void {
    event.stopPropagation(); // Prevent card click
    this.delete.emit(this.prompt.id);
  }

  /**
   * Determine the access level badge class
   */
  getAccessLevelClass(): string {
    return this.prompt.is_public ? 'badge-public' : 'badge-private';
  }
}
