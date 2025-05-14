import { Component, OnInit, OnDestroy, NO_ERRORS_SCHEMA } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Subscription, finalize, Observable, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  Dataset,
  DatasetUpdateRequest,
  Document,
  DatasetDetailResponse,
  DatasetStatus
} from '@ngtx-apps/data-access/models';
import { DatasetService } from '@ngtx-apps/data-access/services';
import { ConfirmationDialogService, AlertService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-dataset-detail',
  templateUrl: './dataset-detail.page.html',
  styleUrls: ['./dataset-detail.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule
  ],
  schemas: [NO_ERRORS_SCHEMA]
})
export class DatasetDetailPage implements OnInit, OnDestroy {
  // Dataset information
  datasetId: string = '';
  dataset: Dataset | null = null;
  documents: Document[] = [];

  // UI state
  isLoading: boolean = true;
  isEditing: boolean = false;
  error: string | null = null;

  // Document preview state
  isPreviewActive: boolean = false;
  currentPreviewDocument: Document | null = null;
  isLoadingPreview: boolean = false;
  documentPreviewContent: string = '';
  documentPreviewHeaders: string[] = [];
  documentPreviewData: any[] = [];
  documentPreviewError: string | null = null;

  // File upload
  selectedFiles: File[] = [];
  uploadProgress: number = 0;
  isUploading: boolean = false;

  // Editing state
  editingDataset: DatasetUpdateRequest = {
    name: '',
    description: '',
    tags: []
  };

  // Available tags for selection
  availableTags: string[] = [
    'support', 'sales', 'marketing', 'technical', 'feedback',
    'queries', 'internal', 'external', 'training', 'evaluation'
  ];

  // Status enums for template
  DatasetStatus = DatasetStatus;

  // Expanded sections in preview
  expandedSections = {
    input: false,
    output: false,
    raw: false
  };

  // Subscriptions
  private subscriptions: Subscription = new Subscription();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private datasetService: DatasetService,
    private confirmationDialogService: ConfirmationDialogService,
    private alertService: AlertService
  ) {}

  ngOnInit(): void {
    // Get dataset ID from route
    this.subscriptions.add(
      this.route.paramMap.subscribe(params => {
        const id = params.get('id');

        if (id) {
          this.datasetId = id;
          this.loadDatasetDetails();
        } else {
          this.router.navigate(['/app/datasets/datasets']);
        }
      })
    );
  }

  ngOnDestroy(): void {
    this.subscriptions.unsubscribe();
  }

  /**
   * Load dataset details including documents
   */
  loadDatasetDetails(): void {
    this.isLoading = true;
    this.error = null;

    this.subscriptions.add(
      this.datasetService.getDataset(this.datasetId).subscribe({
        next: (response: DatasetDetailResponse) => {
          this.dataset = response.dataset;
          this.documents = response.documents || [];
          this.isLoading = false;
        },
        error: (err: any) => {
          console.error('Error loading dataset:', err);
          this.error = 'Failed to load dataset details. Please try again.';
          this.isLoading = false;
        }
      })
    );
  }

  /**
   * Navigate back to datasets list
   */
  goBack(event: Event): void {
    event.preventDefault();
    this.router.navigate(['/app/datasets/datasets']);
  }

  /**
   * Start editing dataset
   */
  startEditing(): void {
    if (!this.dataset) return;

    this.editingDataset = {
      name: this.dataset.name || '',
      description: this.dataset.description || '',
      tags: [...(this.dataset.tags || [])]
    };

    this.isEditing = true;
  }

  /**
   * Toggle tag selection while editing
   */
  toggleTag(tag: string): void {
    if (!this.editingDataset.tags) {
      this.editingDataset.tags = [];
    }

    const index = this.editingDataset.tags.indexOf(tag);
    if (index === -1) {
      this.editingDataset.tags.push(tag);
    } else {
      this.editingDataset.tags.splice(index, 1);
    }
  }

  /**
   * Cancel editing mode
   */
  cancelEditing(): void {
    this.isEditing = false;
  }

  /**
   * Check if changes can be saved
   */
  canSave(): boolean {
    return !!this.editingDataset.name && this.editingDataset.name.trim() !== '';
  }

  /**
   * Save dataset changes
   */
  saveChanges(): void {
    if (!this.canSave() || !this.dataset) return;

    // Create clean update request
    const updateRequest: DatasetUpdateRequest = {
      name: this.editingDataset.name?.trim() || '',
      description: this.editingDataset.description?.trim() || '',
      tags: this.editingDataset.tags || []
    };

    // Update dataset
    this.isLoading = true;

    this.subscriptions.add(
      this.datasetService.updateDataset(this.datasetId, updateRequest)
        .pipe(finalize(() => {
          this.isLoading = false;
          this.isEditing = false;
        }))
        .subscribe({
          next: (updatedDataset) => {
            this.dataset = updatedDataset;
            this.alertService.showAlert({
              show: true,
              message: 'Dataset updated successfully',
              title: 'Success'
            });
          },
          error: (err: any) => {
            console.error('Error updating dataset:', err);
            this.alertService.showAlert({
              show: true,
              message: 'Failed to update dataset. Please try again.',
              title: 'Error'
            });
          }
        })
    );
  }

  /**
   * Delete dataset
   */
  deleteDataset(event: Event): void {
    event.preventDefault();

    this.confirmationDialogService.confirm({
      title: 'Delete Dataset',
      message: 'Are you sure you want to delete this dataset? This action cannot be undone.',
      confirmText: 'Delete',
      cancelText: 'Cancel',
      type: 'danger'
    }).subscribe((confirmed: boolean) => {
      if (confirmed) {
        this.isLoading = true;

        this.subscriptions.add(
          this.datasetService.deleteDataset(this.datasetId)
            .pipe(finalize(() => this.isLoading = false))
            .subscribe({
              next: () => {
                this.alertService.showAlert({
                  show: true,
                  message: 'Dataset deleted successfully',
                  title: 'Success'
                });
                this.router.navigate(['/app/datasets/datasets']);
              },
              error: (err: any) => {
                console.error('Error deleting dataset:', err);
                this.alertService.showAlert({
                  show: true,
                  message: 'Failed to delete dataset. Please try again.',
                  title: 'Error'
                });
              }
            })
        );
      }
    });
  }

  /**
   * Format a date string
   */
  formatDate(dateString: string | undefined): string {
    if (!dateString) return 'N/A';

    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }

  /**
   * Get CSS class for status badge
   */
  getStatusBadgeClass(status: DatasetStatus | string): string {
    if (!status) return '';

    const statusStr = status.toString().toLowerCase();

    switch (statusStr) {
      case DatasetStatus.READY.toLowerCase():
        return 'status-ready';
      case DatasetStatus.PROCESSING.toLowerCase():
        return 'status-processing';
      case DatasetStatus.ERROR.toLowerCase():
        return 'status-error';
      default:
        return '';
    }
  }

  /**
   * Get formatted size of dataset
   */
  get formattedSize(): string {
    if (!this.dataset) return '0 B';

    const bytes = this.dataset.size || 0;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];

    if (bytes === 0) return '0 B';

    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  }

  /**
   * Check if dataset has a document
   */
  hasDocument(): boolean {
    return (this.documents && this.documents.length > 0) ||
           (this.dataset?.metadata?.['meta_info']?.['filename'] != null);
  }

  /**
   * Upload document to dataset
   */
  uploadDocuments(event: Event): void {
    event.preventDefault();

    // Navigate to upload page with existing dataset ID
    this.router.navigate(['/app/datasets/datasets/upload'], {
      queryParams: { datasetId: this.datasetId }
    });
  }

  /**
   * Get document format from file name
   */
  getDocumentFormat(document: Document | null): string {
    if (!document || !document.name) return 'Unknown';

    const filename = document.name.toLowerCase();

    if (filename.endsWith('.csv')) return 'CSV';
    if (filename.endsWith('.txt')) return 'TXT';
    if (filename.endsWith('.json')) return 'JSON';
    if (filename.endsWith('.pdf')) return 'PDF';
    if (filename.endsWith('.docx')) return 'DOCX';

    // Extract extension
    const extension = filename.split('.').pop();
    return extension ? extension.toUpperCase() : 'Unknown';
  }

  /**
   * Preview document content
   */
  previewDocument(document: Document | { id: string; name: string } | null): void {
    if (!document || !document.id) {
      return;
    }

    // Create a Document-like object for files that aren't in the documents array
    const documentToPreview: Document = 'datasetId' in document
      ? document
      : {
          id: document.id,
          datasetId: this.datasetId,
          name: document.name,
          content: '',
          createdAt: new Date().toISOString()
        };

    this.isPreviewActive = true;
    this.currentPreviewDocument = documentToPreview;
    this.isLoadingPreview = true;
    this.documentPreviewError = null;

    // In a real implementation, we would call an API to fetch the document content
    this.getDocumentContent(documentToPreview.id)
      .subscribe({
        next: (content) => {
          this.isLoadingPreview = false;
          // Process the document content based on format
          this.processDocumentContent(content, this.getDocumentFormat(documentToPreview));
        },
        error: (error) => {
          this.isLoadingPreview = false;
          this.documentPreviewError = 'Failed to load document content. Please try again.';
          console.error('Error loading document content:', error);
        }
      });
  }

  /**
   * Get document content from API
   * Note: This is a placeholder that should be replaced with actual API call
   */
  private getDocumentContent(documentId: string): Observable<any> {
    // Simulate API call with a delay
    return of(this.getMockContent(documentId)).pipe(
      catchError(error => {
        console.error('Error fetching document content:', error);
        return of(null);
      })
    );
  }

  /**
   * Process document content based on format
   */
  private processDocumentContent(content: any, format: string): void {
    if (!content) {
      this.documentPreviewError = 'No content available for this document';
      return;
    }

    if (format === 'CSV') {
      try {
        // For CSV, parse the content and extract headers and rows
        const rows = content.split('\n');
        if (rows.length > 0) {
          this.documentPreviewHeaders = rows[0].split(',').map((h: string) => h.trim());

          this.documentPreviewData = [];
          for (let i = 1; i < rows.length && i < 10; i++) { // Limit to 10 rows for preview
            if (rows[i].trim()) {
              const values = rows[i].split(',');
              const rowData: Record<string, string> = {};

              this.documentPreviewHeaders.forEach((header, index) => {
                rowData[header] = values[index]?.trim() || '';
              });

              this.documentPreviewData.push(rowData);
            }
          }
        }
      } catch (e) {
        console.error('Error parsing CSV:', e);
        this.documentPreviewError = 'Error parsing CSV content';
      }
    } else if (format === 'JSON') {
      try {
        // For JSON, format the content for display
        this.documentPreviewContent = typeof content === 'string'
          ? content
          : JSON.stringify(content, null, 2);
      } catch (e) {
        console.error('Error parsing JSON:', e);
        this.documentPreviewError = 'Error parsing JSON content';
      }
    } else {
      // For other formats, display as plain text
      this.documentPreviewContent = content;
    }
  }

  /**
   * Get mock content for document preview (simulated API response)
   */
  private getMockContent(documentId: string): any {
    const format = this.currentPreviewDocument
      ? this.getDocumentFormat(this.currentPreviewDocument).toLowerCase()
      : '';

    if (format === 'csv') {
      return 'query,ground_truth,context\nHow do I reset my password?,Go to login page and click "Forgot Password",Account management\nWhere is my order?,Check order status in your account,Order tracking\nHow to cancel subscription?,Go to account settings, select subscriptions,Subscription management';
    } else if (format === 'json') {
      return JSON.stringify({
        "items": [
          {
            "query": "How do I reset my password?",
            "ground_truth": "Go to login page and click \"Forgot Password\"",
            "context": "Account management"
          },
          {
            "query": "Where is my order?",
            "ground_truth": "Check order status in your account",
            "context": "Order tracking"
          }
        ]
      }, null, 2);
    } else {
      return "Sample text content for document preview.\nThis would contain the actual document content in a real implementation.";
    }
  }

  /**
   * Format JSON for display
   */
  formatJson(json: string): string {
    try {
      const parsed = JSON.parse(json);
      return JSON.stringify(parsed, null, 2);
    } catch (e) {
      return json;
    }
  }

  /**
   * Close document preview
   */
  closeDocumentPreview(): void {
    this.isPreviewActive = false;
    this.currentPreviewDocument = null;
    this.documentPreviewContent = '';
    this.documentPreviewHeaders = [];
    this.documentPreviewData = [];
    this.documentPreviewError = null;
  }

  /**
   * Toggle expanded section in preview
   */
  toggleSection(section: keyof typeof this.expandedSections): void {
    this.expandedSections[section] = !this.expandedSections[section];
  }
}