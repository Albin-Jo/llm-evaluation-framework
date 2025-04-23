/* Path: libs/utils/services/src/lib/notification.service.ts */
import { Injectable, ApplicationRef, ComponentRef, createComponent, EnvironmentInjector, inject } from '@angular/core';
import { NotificationToastComponent, NotificationConfig } from '@ngtx-apps/ui/components';

@Injectable({
  providedIn: 'root'
})
export class NotificationService {
  private toastComponentRef: ComponentRef<NotificationToastComponent> | null = null;
  private appRef = inject(ApplicationRef);
  private environmentInjector = inject(EnvironmentInjector);

  constructor() {
    this.initializeToastComponent();
  }

  private initializeToastComponent(): void {
    // Create the component if it doesn't exist
    if (!this.toastComponentRef) {
      // Create the component dynamically
      this.toastComponentRef = createComponent(NotificationToastComponent, {
        environmentInjector: this.environmentInjector
      });

      // Attach to the DOM
      document.body.appendChild(this.toastComponentRef.location.nativeElement);

      // Attach to the Angular change detection
      this.appRef.attachView(this.toastComponentRef.hostView);
    }
  }

  /**
   * Show a success notification
   */
  success(message: string, title: string = 'Success', duration: number = 5000): number {
    return this.show({
      type: 'success',
      message,
      title,
      duration
    });
  }

  /**
   * Show an error notification
   */
  error(message: string, title: string = 'Error', duration: number = 5000): number {
    return this.show({
      type: 'error',
      message,
      title,
      duration
    });
  }

  /**
   * Show an info notification
   */
  info(message: string, title: string = 'Information', duration: number = 5000): number {
    return this.show({
      type: 'info',
      message,
      title,
      duration
    });
  }

  /**
   * Show a warning notification
   */
  warning(message: string, title: string = 'Warning', duration: number = 5000): number {
    return this.show({
      type: 'warning',
      message,
      title,
      duration
    });
  }

  /**
   * Show a notification with the given configuration
   */
  show(config: NotificationConfig): number {
    this.initializeToastComponent();
    return this.toastComponentRef!.instance.show(config);
  }

  /**
   * Dismiss a notification by id
   */
  dismiss(id: number): void {
    if (this.toastComponentRef) {
      this.toastComponentRef.instance.dismiss(id);
    }
  }

  /**
   * Clear all notifications
   */
  clearAll(): void {
    if (this.toastComponentRef) {
      this.toastComponentRef.instance.clearAll();
    }
  }
}
