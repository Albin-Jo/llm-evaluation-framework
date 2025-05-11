import { Component, EventEmitter, Input, Output, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Dataset, DatasetStatus } from '@ngtx-apps/data-access/models';
import { QracButtonComponent, QracTagButtonComponent } from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-dataset-card',
  standalone: true,
  imports: [CommonModule],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './dataset-card.component.html',
  styleUrls: ['./dataset-card.component.scss']
})
export class DatasetCardComponent {
  @Input() dataset!: Dataset;
  @Output() delete = new EventEmitter<Event>();

  get statusClass(): string {
    switch (this.dataset.status) {
      case DatasetStatus.READY:
        return 'status-ready';
      case DatasetStatus.PROCESSING:
        return 'status-processing';
      case DatasetStatus.ERROR:
        return 'status-error';
      default:
        return '';
    }
  }

  get statusText(): string {
    switch (this.dataset.status) {
      case DatasetStatus.READY:
        return 'Ready';
      case DatasetStatus.PROCESSING:
        return 'Processing';
      case DatasetStatus.ERROR:
        return 'Error';
      default:
        return 'Unknown';
    }
  }

  formatDate(dateString: string): string {
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

  formatSize(bytes?: number): string {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  onDelete(event: Event): void {
    event.stopPropagation(); // Prevent card click
    this.delete.emit(event);
  }
}
