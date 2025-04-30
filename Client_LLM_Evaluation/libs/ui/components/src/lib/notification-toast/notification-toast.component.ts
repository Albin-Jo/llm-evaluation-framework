/* Path: libs/ui/components/src/lib/notification-toast/notification-toast.component.ts */
import { Component, ElementRef, OnDestroy, OnInit, Renderer2 } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, timer } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

export interface NotificationConfig {
  title: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

@Component({
  selector: 'app-notification-toast',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './notification-toast.component.html',
  styleUrls: ['./notification-toast.component.scss']
})
export class NotificationToastComponent implements OnInit, OnDestroy {
  notifications: (NotificationConfig & { id: number, visible: boolean })[] = [];
  private counter = 0;
  private defaultDuration = 5000; // 5 seconds
  private destroy$ = new Subject<void>();

  constructor(private renderer: Renderer2, private el: ElementRef) {}

  ngOnInit(): void {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  show(config: NotificationConfig): number {
    const id = ++this.counter;
    const duration = config.duration || this.defaultDuration;

    // Add new notification
    const notification = {
      ...config,
      id,
      visible: true
    };
    this.notifications.push(notification);

    // Auto dismiss after duration
    timer(duration)
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        this.dismiss(id);
      });

    return id;
  }

  dismiss(id: number): void {
    // Find notification and mark it as not visible
    const index = this.notifications.findIndex(n => n.id === id);
    if (index !== -1) {
      this.notifications[index].visible = false;

      // Remove notification from array after animation completes
      setTimeout(() => {
        this.notifications = this.notifications.filter(n => n.id !== id);
      }, 300); // Match the animation duration
    }
  }

  clearAll(): void {
    // Mark all as not visible
    this.notifications.forEach(notification => {
      notification.visible = false;
    });

    // Remove all after animation
    setTimeout(() => {
      this.notifications = [];
    }, 300);
  }

  getIconClass(type: string): string {
    switch (type) {
      case 'success':
        return 'icon-success';
      case 'error':
        return 'icon-error';
      case 'warning':
        return 'icon-warning';
      case 'info':
      default:
        return 'icon-info';
    }
  }
}
