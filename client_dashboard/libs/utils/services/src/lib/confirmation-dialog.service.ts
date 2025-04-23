/* Path: libs/utils/services/src/lib/confirmation-dialog.service.ts */
import { Injectable, ApplicationRef, ComponentRef, createComponent, EnvironmentInjector, inject } from '@angular/core';
import { ConfirmationDialogComponent, ConfirmationDialogConfig } from '@ngtx-apps/ui/components';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ConfirmationDialogService {
  private dialogComponentRef: ComponentRef<ConfirmationDialogComponent> | null = null;
  private appRef = inject(ApplicationRef);
  private environmentInjector = inject(EnvironmentInjector);

  constructor() {}

  private createDialogComponent(): ComponentRef<ConfirmationDialogComponent> {
    // Create the component dynamically
    const componentRef = createComponent(ConfirmationDialogComponent, {
      environmentInjector: this.environmentInjector
    });

    // Attach to the DOM
    document.body.appendChild(componentRef.location.nativeElement);

    // Attach to the Angular change detection
    this.appRef.attachView(componentRef.hostView);

    return componentRef;
  }

  /**
   * Show a confirmation dialog and return an Observable that resolves when the user makes a choice
   * @param config Configuration for the confirmation dialog
   * @returns Observable that resolves to true if confirmed, false if cancelled
   */
  confirm(config: ConfirmationDialogConfig): Observable<boolean> {
    // Clean up any existing dialog
    this.destroyDialog();

    // Create a new dialog
    this.dialogComponentRef = this.createDialogComponent();
    const instance = this.dialogComponentRef.instance;

    // Set the configuration
    instance.config = {
      ...{
        title: 'Confirm Action',
        message: 'Are you sure you want to proceed?',
        confirmText: 'Confirm',
        cancelText: 'Cancel',
        type: 'warning'
      },
      ...config
    };

    // Show the dialog
    instance.show();

    // Return an Observable that resolves when the user makes a choice
    return new Observable<boolean>(observer => {
      const confirmSub = instance.confirm.subscribe(() => {
        observer.next(true);
        observer.complete();
        this.destroyDialog();
      });

      const cancelSub = instance.cancel.subscribe(() => {
        observer.next(false);
        observer.complete();
        this.destroyDialog();
      });

      const closeSub = instance.close.subscribe(() => {
        observer.next(false);
        observer.complete();
        this.destroyDialog();
      });

      // Clean up subscriptions when the observer is unsubscribed
      return () => {
        confirmSub.unsubscribe();
        cancelSub.unsubscribe();
        closeSub.unsubscribe();
      };
    });
  }

  /**
   * Convenience method for delete confirmations
   */
  confirmDelete(entityName: string = 'item'): Observable<boolean> {
    return this.confirm({
      title: `Delete ${entityName}`,
      message: `Are you sure you want to delete this ${entityName.toLowerCase()}? This action cannot be undone.`,
      confirmText: 'Delete',
      cancelText: 'Cancel',
      type: 'danger'
    });
  }

  /**
   * Clean up the dialog component
   */
  private destroyDialog(): void {
    if (this.dialogComponentRef) {
      this.appRef.detachView(this.dialogComponentRef.hostView);
      this.dialogComponentRef.destroy();
      this.dialogComponentRef = null;
    }
  }
}
