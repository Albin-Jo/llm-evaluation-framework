/* Path: libs/ui/components/src/lib/confirmation-dialog/confirmation-dialog.component.ts */
import { Component, ElementRef, EventEmitter, Input, OnDestroy, OnInit, Output, Renderer2 } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface ConfirmationDialogConfig {
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  type?: 'warning' | 'danger' | 'info' | 'success';
}

@Component({
  selector: 'app-confirmation-dialog',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './confirmation-dialog.component.html',
  styleUrls: ['./confirmation-dialog.component.scss']
})
export class ConfirmationDialogComponent implements OnInit, OnDestroy {
  @Input() config: ConfirmationDialogConfig = {
    title: 'Confirm Action',
    message: 'Are you sure you want to proceed?',
    confirmText: 'Confirm',
    cancelText: 'Cancel',
    type: 'warning'
  };

  @Output() confirm = new EventEmitter<void>();
  @Output() cancel = new EventEmitter<void>();
  @Output() close = new EventEmitter<void>();

  visible = false;
  private unlistenFn: (() => void) | null = null;

  constructor(private renderer: Renderer2, private el: ElementRef) {}

  ngOnInit(): void {
    // Prevent background scrolling when dialog is open
    this.unlistenFn = this.renderer.listen('document', 'keydown', (event: KeyboardEvent) => {
      if (event.key === 'Escape' && this.visible) {
        this.onCancel();
      }
    });
  }

  ngOnDestroy(): void {
    if (this.unlistenFn) {
      this.unlistenFn();
    }
  }

  show(): void {
    this.visible = true;
    // Prevent background scrolling
    this.renderer.setStyle(document.body, 'overflow', 'hidden');
  }

  hide(): void {
    this.visible = false;
    // Restore background scrolling
    this.renderer.removeStyle(document.body, 'overflow');
  }

  onConfirm(): void {
    this.confirm.emit();
    this.hide();
  }

  onCancel(): void {
    this.cancel.emit();
    this.hide();
  }

  onClose(): void {
    this.close.emit();
    this.hide();
  }

  onOverlayClick(event: MouseEvent): void {
    // Only close if the overlay background was clicked, not the dialog itself
    if (event.target === event.currentTarget) {
      this.onCancel();
    }
  }

  getIconClass(): string {
    switch (this.config.type) {
      case 'danger':
        return 'icon-danger';
      case 'success':
        return 'icon-success';
      case 'info':
        return 'icon-info';
      case 'warning':
      default:
        return 'icon-warning';
    }
  }
}
