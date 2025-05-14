import { Component, OnInit, OnDestroy, NO_ERRORS_SCHEMA } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Subscription, finalize, Observable, of, throwError } from 'rxjs';
import { catchError, tap, switchMap } from 'rxjs/operators';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  Dataset,
  DatasetUpdateRequest,
  Document,
  DatasetDetailResponse,
  DatasetStatus,
} from '@ngtx-apps/data-access/models';
import {
  DatasetService,
  DatasetContentService,
} from '@ngtx-apps/data-access/services';
import {
  ConfirmationDialogService,
  AlertService,
} from '@ngtx-apps/utils/services';

// Define an enum for section types to fix the TypeScript error
enum ExpandedSection {
  INPUT = 'input',
  OUTPUT = 'output',
  RAW = 'raw',
}

@Component({
  selector: 'app-dataset-detail',
  templateUrl: './dataset-detail.page.html',
  styleUrls: ['./dataset-detail.page.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule],
  schemas: [NO_ERRORS_SCHEMA],
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
  isDeleting: boolean = false;

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
    tags: [],
  };

  // Available tags for selection
  availableTags: string[] = [
    'support',
    'sales',
    'marketing',
    'technical',
    'feedback',
    'queries',
    'internal',
    'external',
    'training',
    'evaluation',
  ];

  // Status enums for template
  DatasetStatus = DatasetStatus;

  // Expanded sections enum for the template
  ExpandedSection = ExpandedSection;

  // Expanded sections in preview - using the enum now
  expandedSections: { [key in ExpandedSection]: boolean } = {
    [ExpandedSection.INPUT]: false,
    [ExpandedSection.OUTPUT]: false,
    [ExpandedSection.RAW]: false,
  };

  // Subscriptions
  private subscriptions: Subscription = new Subscription();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private datasetService: DatasetService,
    private datasetContentService: DatasetContentService,
    private confirmationDialogService: ConfirmationDialogService,
    private alertService: AlertService
  ) {}

  ngOnInit(): void {
    // Get dataset ID from route
    this.subscriptions.add(
      this.route.paramMap.subscribe((params) => {
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
        },
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
      tags: [...(this.dataset.tags || [])],
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
      tags: this.editingDataset.tags || [],
    };

    // Update dataset
    this.isLoading = true;

    this.subscriptions.add(
      this.datasetService
        .updateDataset(this.datasetId, updateRequest)
        .pipe(
          finalize(() => {
            this.isLoading = false;
            this.isEditing = false;
          })
        )
        .subscribe({
          next: (updatedDataset) => {
            this.dataset = updatedDataset;
            this.alertService.showAlert({
              show: true,
              message: 'Dataset updated successfully',
              title: 'Success',
            });
          },
          error: (err: any) => {
            console.error('Error updating dataset:', err);
            this.alertService.showAlert({
              show: true,
              message: 'Failed to update dataset. Please try again.',
              title: 'Error',
            });
          },
        })
    );
  }

  /**
   * Delete dataset
   */
  deleteDataset(event: Event): void {
    event.preventDefault();

    // Prevent multiple deletion requests
    if (this.isDeleting) return;

    this.confirmationDialogService
      .confirm({
        title: 'Delete Dataset',
        message:
          'Are you sure you want to delete this dataset? This action cannot be undone.',
        confirmText: 'Delete',
        cancelText: 'Cancel',
        type: 'danger',
      })
      .subscribe((confirmed: boolean) => {
        if (confirmed) {
          this.isDeleting = true;
          this.isLoading = true;

          this.subscriptions.add(
            this.datasetService
              .deleteDataset(this.datasetId)
              .pipe(
                finalize(() => {
                  this.isLoading = false;
                  this.isDeleting = false;
                })
              )
              .subscribe({
                next: () => {
                  this.alertService.showAlert({
                    show: true,
                    message: 'Dataset deleted successfully',
                    title: 'Success',
                  });
                  // Ensure navigation happens after alert is shown
                  setTimeout(() => {
                    this.router.navigate(['/app/datasets/datasets']);
                  }, 100);
                },
                error: (err: any) => {
                  console.error('Error deleting dataset:', err);
                  this.alertService.showAlert({
                    show: true,
                    message: 'Failed to delete dataset. Please try again.',
                    title: 'Error',
                  });
                },
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
      day: 'numeric',
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
    return (
      (this.documents && this.documents.length > 0) ||
      this.dataset?.metadata?.['meta_info']?.['filename'] != null
    );
  }

  /**
   * Upload document to dataset
   */
  uploadDocuments(event: Event): void {
    event.preventDefault();

    // Navigate to upload page with existing dataset ID
    this.router.navigate(['/app/datasets/datasets/upload'], {
      queryParams: { datasetId: this.datasetId },
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
   * Preview document content - Updated to use dataset content service
   */
  previewDocument(
    document: Document | { id: string; name: string } | null
  ): void {
    if (!document || !document.id) {
      return;
    }

    // Create a Document-like object for files that aren't in the documents array
    const documentToPreview: Document =
      'datasetId' in document
        ? document
        : {
            id: document.id,
            datasetId: this.datasetId,
            name: document.name,
            content: '',
            createdAt: new Date().toISOString(),
          };

    this.isPreviewActive = true;
    this.currentPreviewDocument = documentToPreview;
    this.isLoadingPreview = true;
    this.documentPreviewError = null;
    this.documentPreviewContent = '';
    this.documentPreviewHeaders = [];
    this.documentPreviewData = [];

    // Call the document content API with the dataset content service
    this.subscriptions.add(
      this.datasetContentService
        .getDocumentContent(this.datasetId, documentToPreview.id)
        .subscribe({
          next: (response) => {
            this.isLoadingPreview = false;

            // SPECIAL CASE: Check if response is a direct array (not wrapped in response object)
            if (Array.isArray(response)) {
              this.handleDirectArrayResponse(response);
              return;
            }

            // Handle error from API
            if (response && response.error) {
              console.error('Error in API response:', response.error);
              this.documentPreviewError = response.error;
              return;
            }

            // Direct approach for tabular data from the API
            if (
              response &&
              response.headers &&
              response.headers.length > 0 &&
              response.rows &&
              response.rows.length > 0
            ) {
              this.documentPreviewHeaders = response.headers;
              this.documentPreviewData = response.rows;

              // Still set content for raw view option
              if (response.content) {
                this.documentPreviewContent =
                  typeof response.content === 'string'
                    ? response.content
                    : JSON.stringify(response.content, null, 2);
              }
              return;
            }

            // If no structured data but content exists
            if (response && response.content) {
              // Handle string or object content
              const contentToProcess =
                typeof response.content === 'string'
                  ? response.content
                  : JSON.stringify(response.content, null, 2);

              this.documentPreviewContent = contentToProcess;

              // Try to parse as JSON if content exists but no headers/rows
              try {
                // Parse if string, otherwise use directly
                const parsedContent =
                  typeof response.content === 'string'
                    ? JSON.parse(response.content)
                    : response.content;

                // Handle array data
                if (Array.isArray(parsedContent)) {
                  this.processArrayContent(parsedContent);
                }
              } catch (e) {
                console.warn('Content is not JSON or failed to parse:', e);
                // Keep as text content - already set above
              }
            } else if (
              typeof response === 'object' &&
              Object.keys(response).length > 0
            ) {
              // If response is an object but doesn't have the expected structure

              // Treat entire response as content
              this.documentPreviewContent = JSON.stringify(response, null, 2);

              // Check if any properties are arrays that could be displayed as tabular data
              for (const key in response) {
                if (
                  Array.isArray(response['key']) &&
                  response['key'].length > 0
                ) {
                  this.processArrayContent(response['key']);
                  break;
                }
              }
            } else {
              console.warn(
                'No content received from API or unrecognized format'
              );
              this.documentPreviewError =
                'No content available for this document';
            }

            // If we still don't have headers/content, show error
            if (
              !this.documentPreviewContent &&
              this.documentPreviewHeaders.length === 0
            ) {
              this.documentPreviewError =
                'No viewable content available for this document';
            }
          },
          error: (error) => {
            console.error('Error loading document content:', error);
            this.isLoadingPreview = false;
            this.documentPreviewError =
              'Failed to load document content. Please try again.';
          },
        })
    );
  }

  /**
   * Handle API response that is a direct array
   */
  private handleDirectArrayResponse(arrayData: any[]): void {
    // Set the raw content for text view
    this.documentPreviewContent = JSON.stringify(arrayData, null, 2);

    // Process for tabular view if it's an array of objects
    if (
      arrayData.length > 0 &&
      typeof arrayData[0] === 'object' &&
      arrayData[0] !== null
    ) {
      this.processArrayContent(arrayData);
    }
  }

  /**
   * Process array content into tabular format
   */
  private processArrayContent(arrayData: any[]): void {
    if (
      arrayData.length === 0 ||
      typeof arrayData[0] !== 'object' ||
      arrayData[0] === null
    ) {
      return;
    }

    // Extract all unique keys across all objects to ensure we have all possible columns
    const allKeys = new Set<string>();

    arrayData.forEach((item) => {
      if (item && typeof item === 'object') {
        Object.keys(item).forEach((key) => allKeys.add(key));
      }
    });

    if (allKeys.size > 0) {
      this.documentPreviewHeaders = Array.from(allKeys);
      this.documentPreviewData = arrayData;
    }
  }

  /**
   * For development/testing only - can be removed in production
   */
  private fallbackToMockData = false; // Set to true only during development if API is not ready

  /**
   * Load mock data as fallback during development - can be removed in production
   */
  private loadMockDataForPreview(documentToPreview: Document): void {
    const format = this.getDocumentFormat(documentToPreview).toLowerCase();

    if (format === 'csv') {
      const mockCsvContent =
        'query,ground_truth,context\nHow do I reset my password?,Go to login page and click "Forgot Password",Account management\nWhere is my order?,Check order status in your account,Order tracking\nHow to cancel subscription?,Go to account settings, select subscriptions,Subscription management';
      this.documentPreviewContent = mockCsvContent;

      // Parse CSV for tabular display
      const parsedCsv =
        this.datasetContentService.parseCSVContent(mockCsvContent);
      this.documentPreviewHeaders = parsedCsv.headers;
      this.documentPreviewData = parsedCsv.data;
    } else if (format === 'json') {
      const mockJsonContent = {
        items: [
          {
            query: 'How do I reset my password?',
            ground_truth: 'Go to login page and click "Forgot Password"',
            context: 'Account management',
          },
          {
            query: 'Where is my order?',
            ground_truth: 'Check order status in your account',
            context: 'Order tracking',
          },
        ],
      };

      this.documentPreviewContent = JSON.stringify(mockJsonContent, null, 2);

      // Try to extract tabular data
      if (mockJsonContent.items && Array.isArray(mockJsonContent.items)) {
        const headers = new Set<string>();
        mockJsonContent.items.forEach((item) => {
          Object.keys(item).forEach((key) => headers.add(key));
        });

        this.documentPreviewHeaders = Array.from(headers);
        this.documentPreviewData = mockJsonContent.items;
      }
    } else {
      this.documentPreviewContent =
        'Sample text content for document preview.\nThis would contain the actual document content in a real implementation.';
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
   * Fixed to use enum instead of 'this' reference in parameter type
   */
  toggleSection(section: ExpandedSection): void {
    this.expandedSections[section] = !this.expandedSections[section];
  }
}
