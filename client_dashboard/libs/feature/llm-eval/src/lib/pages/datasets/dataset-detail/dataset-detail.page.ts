/* Path: libs/feature/llm-eval/src/lib/pages/datasets/dataset-detail/dataset-detail.page.ts */
import { Component, OnInit, OnDestroy, NO_ERRORS_SCHEMA } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Subscription, finalize } from 'rxjs';
import { Location } from '@angular/common';
import {
  Dataset,
  DatasetUpdateRequest,
  Document,
  DatasetDetailResponse
} from '@ngtx-apps/data-access/models';
import { DatasetService } from '@ngtx-apps/data-access/services';
import { ConfirmationDialogService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-dataset-detail',
  templateUrl: './dataset-detail.page.html',
  styleUrls: ['./dataset-detail.page.scss'],
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

  // Document preview
  previewingDocument: boolean = false;
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

  // Subscriptions
  private subscriptions: Subscription = new Subscription();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private datasetService: DatasetService,
    private location: Location,
    private confirmationService: ConfirmationDialogService
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
          this.router.navigate(['/datasets']);
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

    this.subscriptions.add(
      this.datasetService.getDataset(this.datasetId).subscribe({
        next: (response: DatasetDetailResponse) => {
          console.log('Dataset detail response:', response);
          this.dataset = response.dataset;
          this.documents = response.documents || [];

          // Reset error state
          this.error = null;
        },
        error: (err: any) => {
          console.error('Error loading dataset:', err);
          this.error = 'Failed to load dataset details. Please try again.';
          this.showErrorMessage('Error', 'Failed to load dataset details');
        },
        complete: () => {
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
    this.location.back();
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
  get statusClass(): string {
    if (!this.dataset) return '';

    switch (this.dataset.status?.toLowerCase()) {
      case 'ready':
        return 'status-ready';
      case 'processing':
        return 'status-processing';
      case 'error':
        return 'status-error';
      default:
        return '';
    }
  }

  /**
   * Get display text for status
   */
  get statusText(): string {
    return this.dataset?.status || 'Unknown';
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
   * Get average document length in tokens (placeholder calculation)
   */
  getAverageDocLength(): number {
    if (!this.dataset || !this.dataset.documentCount || this.dataset.documentCount === 0) {
      return 0;
    }

    // This is a placeholder. In a real application, this would come from the API
    return Math.round((this.dataset.size || 0) / (this.dataset.documentCount * 4)); // Average 4 bytes per token
  }

  /**
   * Get file format distribution (placeholder implementation)
   */
  getFileFormatDistribution(): string {
    if (this.documents.length === 0) {
      return 'No documents';
    }

    // Group documents by file extension
    const formats: Record<string, number> = {};

    this.documents.forEach(doc => {
      const format = this.getDocumentFormat(doc);
      formats[format] = (formats[format] || 0) + 1;
    });

    // Convert to readable string
    return Object.entries(formats)
      .map(([format, count]) => `${format}: ${count}`)
      .join(', ');
  }

  /**
   * Get document format based on filename
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
   * Get document size formatted
   */
  getDocumentSize(document: Document): string {
    // Checking for size safely
    const docSize = document?.metadata?.size || 0;

    const bytes = docSize;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];

    if (bytes === 0) return '0 B';

    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  }

  /**
   * Open document preview modal
   */
  previewDocument(document: Document): void {
    if (!document || !document.id || !this.datasetId) {
      this.showErrorMessage('Error', 'Cannot preview document: Missing document information');
      return;
    }

    this.previewingDocument = true;
    this.currentPreviewDocument = document;
    this.isLoadingPreview = true;
    this.documentPreviewError = null;

    // Get the document format to determine how to handle the preview
    const documentFormat = this.getDocumentFormat(document);

    // Simple timeout-based preview for now
    // In a real implementation, you would call an API endpoint
    setTimeout(() => {
      this.isLoadingPreview = false;

      if (documentFormat === 'CSV') {
        // Example CSV data
        this.documentPreviewHeaders = ['id', 'name', 'email', 'message'];
        this.documentPreviewData = [
          { id: 1, name: 'John Doe', email: 'john@example.com', message: 'I need help with my account' },
          { id: 2, name: 'Jane Smith', email: 'jane@example.com', message: 'How do I reset my password?' }
        ];
      } else if (documentFormat === 'TXT' || documentFormat === 'JSON') {
        // Example text content
        this.documentPreviewContent = documentFormat === 'JSON'
          ? '{\n  "id": 1,\n  "name": "Sample Document",\n  "content": "This is a sample content"\n}'
          : 'This is a sample text document content.\nLine 2 of the document.\nLine 3 with more text.';
      } else {
        this.documentPreviewError = 'Preview not available for this file type';
      }
    }, 800);
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
   * Close document preview modal
   */
  closeDocumentPreview(): void {
    this.previewingDocument = false;
    this.currentPreviewDocument = null;
    this.documentPreviewContent = '';
    this.documentPreviewHeaders = [];
    this.documentPreviewData = [];
    this.documentPreviewError = null;
  }

  /**
   * Download current preview document
   */
  downloadDocument(): void {
    if (!this.currentPreviewDocument) return;

    // In a real application, this would call an API endpoint to download the file
    this.showInfoMessage('Download', 'Document download started...');

    // Simulate a download
    setTimeout(() => {
      this.showSuccessMessage('Success', 'Document downloaded successfully');
    }, 1500);
  }

  /**
   * Start editing dataset
   */
  startEditing(): void {
    if (!this.dataset) return;

    this.editingDataset = {
      name: this.dataset.name,
      description: this.dataset.description || '',
      tags: [...(this.dataset.tags || [])]
    };

    this.isEditing = true;
  }

  /**
   * Cancel editing
   */
  cancelEditing(): void {
    this.isEditing = false;
  }

  /**
   * Toggle tag selection
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
            this.showSuccessMessage('Success', 'Dataset updated successfully');
          },
          error: (err: any) => {
            console.error('Error updating dataset:', err);
            this.showErrorMessage('Error', 'Failed to update dataset');
          }
        })
    );
  }

  /**
   * Delete dataset
   */
  deleteDataset(event: Event): void {
    event.preventDefault();

    // Show confirmation dialog
    this.confirmationService.confirm({
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
                this.showSuccessMessage('Success', 'Dataset deleted successfully');
                this.router.navigate(['/datasets']);
              },
              error: (err: any) => {
                console.error('Error deleting dataset:', err);
                this.showErrorMessage('Error', 'Failed to delete dataset');
              }
            })
        );
      }
    });
  }

  /**
   * Open file upload dialog
   */
  uploadDocuments(event: Event): void {
    event.preventDefault();

    // Create a hidden file input and trigger click
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.accept = '.csv,.txt,.json,.pdf,.docx';

    fileInput.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      if (target.files && target.files.length > 0) {
        this.handleFileSelection(target.files);
      }
    };

    fileInput.click();
  }

  /**
   * Handle file selection
   */
  private handleFileSelection(files: FileList): void {
    this.selectedFiles = Array.from(files);
    this.uploadSelectedFiles();
  }

  /**
   * Upload selected files
   */
  private uploadSelectedFiles(): void {
    if (this.selectedFiles.length === 0) return;

    this.isUploading = true;
    this.uploadProgress = 0;

    // Upload interval simulation (for demo)
    const interval = setInterval(() => {
      this.uploadProgress += 10;
      if (this.uploadProgress >= 100) {
        clearInterval(interval);
        this.completeUpload();
      }
    }, 300);

    // In a real application, call the API
    this.subscriptions.add(
      this.datasetService.uploadDocumentsToDataset(this.datasetId, this.selectedFiles)
        .subscribe({
          next: (updatedDataset) => {
            // Clear interval
            clearInterval(interval);
            this.uploadProgress = 100;

            // Update dataset and fetch updated document list
            this.dataset = updatedDataset;
            this.loadDatasetDetails();

            this.showSuccessMessage('Success', 'Documents uploaded successfully');
            this.completeUpload();
          },
          error: (err: any) => {
            // Clear interval
            clearInterval(interval);

            console.error('Error uploading documents:', err);
            this.showErrorMessage('Error', 'Failed to upload documents');
            this.completeUpload();
          }
        })
    );
  }

  /**
   * Complete upload process
   */
  private completeUpload(): void {
    this.isUploading = false;
    this.selectedFiles = [];
    this.uploadProgress = 0;
  }

  /**
   * Confirm document deletion
   */
  confirmDeleteDocument(event: Event, documentId: string): void {
    event.preventDefault();

    this.confirmationService.confirm({
      title: 'Delete Document',
      message: 'Are you sure you want to delete this document? This action cannot be undone.',
      confirmText: 'Delete',
      cancelText: 'Cancel',
      type: 'danger'
    }).subscribe((confirmed: boolean) => {
      if (confirmed) {
        this.showInfoMessage('Info', 'Document deletion initiated');

        // Update local document list (in a real app would call API)
        this.documents = this.documents.filter(doc => doc.id !== documentId);

        // If dataset information exists, decrement document count
        if (this.dataset && this.dataset.documentCount !== undefined) {
          this.dataset.documentCount = Math.max(0, this.dataset.documentCount - 1);
        }

        this.showSuccessMessage('Success', 'Document deleted successfully');
      }
    });
  }

  // Toast message helpers
  private showSuccessMessage(title: string, message: string): void {
    console.log(`Success: ${title} - ${message}`);
    // In a real implementation, this would use a toast service
    // this.toastService.success(title, message);
  }

  private showErrorMessage(title: string, message: string): void {
    console.error(`Error: ${title} - ${message}`);
    // In a real implementation, this would use a toast service
    // this.toastService.error(title, message);
  }

  private showInfoMessage(title: string, message: string): void {
    console.log(`Info: ${title} - ${message}`);
    // In a real implementation, this would use a toast service
    // this.toastService.info(title, message);
  }
}