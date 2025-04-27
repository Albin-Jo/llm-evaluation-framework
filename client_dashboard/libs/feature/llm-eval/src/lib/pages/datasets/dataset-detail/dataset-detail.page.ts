/* Path: libs/feature/llm-eval/src/lib/pages/datasets/dataset-detail/dataset-detail.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Subject, takeUntil, switchMap, of, catchError, BehaviorSubject } from 'rxjs';
import {
  Dataset,
  DatasetStatus,
  DatasetUpdateRequest,
  Document
} from '@ngtx-apps/data-access/models';
import { DatasetService } from '@ngtx-apps/data-access/services';
import { AlertService } from '@ngtx-apps/utils/services';

interface TabDefinition {
  id: string;
  label: string;
}

@Component({
  selector: 'app-dataset-detail',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './dataset-detail.page.html',
  styleUrls: ['./dataset-detail.page.scss']
})
export class DatasetDetailPage implements OnInit, OnDestroy {
  // Dataset data
  dataset: Dataset | null = null;
  documents: Document[] = [];

  // UI state
  isLoading = true;
  isEditing = false;
  isSaving = false;
  errorMessage: string | null = null;

  // Tab navigation
  activeTab = 'overview';
  tabs: TabDefinition[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'documents', label: 'Documents' },
    { id: 'statistics', label: 'Statistics' },
    { id: 'settings', label: 'Settings' }
  ];

  // Editing state
  editingDataset: Partial<Dataset> = {};

  // Documents list state
  documentSearchQuery = '';
  documentFormatFilter = '';
  documentSortBy = 'name';
  documentSortDirection: 'asc' | 'desc' = 'asc';
  filteredDocuments: Document[] = [];
  isLoadingDocuments = false;

  // Pagination
  itemsPerPage = 10;
  currentDocumentsPage = 1;
  totalDocumentsPages = 1;

  // Document Preview Modal
  previewingDocument = false;
  currentPreviewDocument: Document | null = null;
  isLoadingPreview = false;
  documentPreviewContent = '';
  documentPreviewError: string | null = null;
  documentPreviewHeaders: string[] = [];
  documentPreviewData: any[] = [];

  // Available Tags and Formats
  availableTags = [
    'documentation',
    'technical',
    'marketing',
    'support',
    'knowledge-base',
    'training',
    'api',
    'legal'
  ];

  availableFormats: string[] = [];

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private datasetService: DatasetService,
    private alertService: AlertService
  ) {}

  ngOnInit(): void {
    this.loadDatasetDetails();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Load dataset details from the API
   */
  loadDatasetDetails(): void {
    this.isLoading = true;
    this.errorMessage = null;

    this.route.paramMap.pipe(
      switchMap(params => {
        const id = params.get('id');

        if (!id) {
          this.router.navigate(['app/datasets/datasets']);
          return of(null);
        }

        return this.datasetService.getDataset(id).pipe(
          catchError(error => {
            this.errorMessage = 'Failed to load dataset. Please try again.';
            console.error('Error loading dataset:', error);
            return of(null);
          })
        );
      }),
      takeUntil(this.destroy$)
    ).subscribe({
      next: (response) => {
        if (response) {
          this.dataset = response.dataset;
          this.documents = response.documents || [];

          if (this.dataset) {
            // Extract unique formats
            this.availableFormats = Array.from(
              new Set(this.documents.map(doc => this.getDocumentFormat(doc)))
            ).filter(format => format);

            // Initialize filtered documents
            this.filterDocuments();
          }
        }
        this.isLoading = false;
      },
      error: (error) => {
        this.alertService.showAlert({
          show: true,
          message: 'Failed to load dataset details. Please try again.',
          title: 'Error'
        });
        this.isLoading = false;
        console.error('Unexpected error:', error);
      }
    });
  }

  /**
   * Set active tab
   */
  setActiveTab(tabId: string): void {
    this.activeTab = tabId;

    // Reset document pagination when switching to documents tab
    if (tabId === 'documents') {
      this.currentDocumentsPage = 1;
      this.filterDocuments();
    }
  }

  /**
   * Start editing dataset
   */
  startEditing(): void {
    if (!this.dataset) return;

    this.editingDataset = {
      ...this.dataset,
      tags: [...(this.dataset.tags || [])]
    };

    this.isEditing = true;
  }

  /**
   * Cancel editing
   */
  cancelEditing(): void {
    this.isEditing = false;
    this.editingDataset = {};
  }

  /**
   * Toggle tag selection during editing
   */
  toggleTag(tag: string): void {
    if (!this.editingDataset.tags) {
      this.editingDataset.tags = [];
    }

    if (this.editingDataset.tags.includes(tag)) {
      this.editingDataset.tags = this.editingDataset.tags.filter(t => t !== tag);
    } else {
      this.editingDataset.tags = [...this.editingDataset.tags, tag];
    }
  }

  /**
   * Check if save is allowed
   */
  canSave(): boolean {
    return !!(this.editingDataset.name && this.editingDataset.name.trim());
  }

  /**
   * Save dataset changes
   */
  saveChanges(): void {
    if (!this.canSave() || !this.dataset) return;

    this.isSaving = true;

    const updateRequest: DatasetUpdateRequest = {
      name: this.editingDataset.name,
      description: this.editingDataset.description,
      tags: this.editingDataset.tags
    };

    this.datasetService.updateDataset(this.dataset.id, updateRequest)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.dataset = response;
          this.alertService.showAlert({
            show: true,
            message: 'Dataset updated successfully',
            title: 'Success'
          });
          this.isSaving = false;
          this.isEditing = false;
        },
        error: (error) => {
          this.alertService.showAlert({
            show: true,
            message: 'Failed to update dataset. Please try again.',
            title: 'Error'
          });
          this.isSaving = false;
          console.error('Error updating dataset:', error);
        }
      });
  }

  /**
   * Delete dataset with confirmation
   */
  deleteDataset(event?: Event): void {
    if (event) {
      event.preventDefault();
    }

    if (!this.dataset) return;

    if (confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
      this.datasetService.deleteDataset(this.dataset.id)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: () => {
            this.alertService.showAlert({
              show: true,
              message: 'Dataset deleted successfully',
              title: 'Success'
            });
            this.router.navigate(['app/datasets/datasets']);
          },
          error: (error) => {
            this.alertService.showAlert({
              show: true,
              message: 'Failed to delete dataset. Please try again.',
              title: 'Error'
            });
            console.error('Error deleting dataset:', error);
          }
        });
    }
  }

  /**
   * Get recent documents for overview tab
   */
  getRecentDocuments(): Document[] {
    return this.documents.slice(0, 5);
  }

  /**
   * Format file size for a document
   */
  getDocumentSize(document: Document | null): string {
    if (!document || !document.metadata?.size) return 'N/A';

    const bytes = document.metadata.size as number;
    return this.formatFileSize(bytes);
  }

  /**
   * Get format/type of a document
   */
  getDocumentFormat(document: Document | null): string {
    if (!document) return '';

    if (document.metadata?.format) {
      return document.metadata.format as string;
    }

    // Try to determine format from filename
    const name = document.name || '';
    const extension = name.split('.').pop()?.toUpperCase() || '';

    return extension || 'Unknown';
  }

  /**
   * Get average document length
   */
  getAverageDocLength(): string {
    if (!this.documents || this.documents.length === 0) return '0';

    // Try to get average token count from metadata
    let totalTokens = 0;
    let documentsWithTokens = 0;

    this.documents.forEach(doc => {
      if (doc.metadata?.token_count) {
        totalTokens += doc.metadata.token_count as number;
        documentsWithTokens++;
      }
    });

    if (documentsWithTokens > 0) {
      return Math.round(totalTokens / documentsWithTokens).toString();
    }

    // Fallback: estimate based on content length
    const totalChars = this.documents.reduce((sum, doc) => sum + (doc.content?.length || 0), 0);
    const avgChars = totalChars / this.documents.length;

    // Rough estimate: ~4 chars per token
    return Math.round(avgChars / 4).toString();
  }

  /**
   * Get file format distribution
   */
  getFileFormatDistribution(): string {
    if (!this.documents || this.documents.length === 0) return 'N/A';

    const formatCounts: Record<string, number> = {};

    this.documents.forEach(doc => {
      const format = this.getDocumentFormat(doc);
      formatCounts[format] = (formatCounts[format] || 0) + 1;
    });

    // Sort by count and format string
    const formats = Object.entries(formatCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2) // Take top 2
      .map(([format, count]) => {
        const percentage = Math.round((count / this.documents.length) * 100);
        return `${format} (${percentage}%)`;
      });

    return formats.join(', ');
  }

  /**
   * Navigate to document upload
   */
  uploadDocuments(event: Event): void {
    event.preventDefault();
    if (!this.dataset) return;

    // Navigate to the upload page with the dataset ID
    this.router.navigate(['app/datasets/datasets/upload'], {
      queryParams: { datasetId: this.dataset.id }
    });
  }

  /**
   * Filter documents based on search and filters
   */
  filterDocuments(): void {
    if (!this.documents) {
      this.filteredDocuments = [];
      this.updatePagination();
      return;
    }

    let result = [...this.documents];

    // Apply search filter
    if (this.documentSearchQuery) {
      const query = this.documentSearchQuery.toLowerCase();
      result = result.filter(doc =>
        doc.name.toLowerCase().includes(query)
      );
    }

    // Apply format filter
    if (this.documentFormatFilter) {
      result = result.filter(doc =>
        this.getDocumentFormat(doc) === this.documentFormatFilter
      );
    }

    // Apply sorting
    result.sort((a, b) => {
      let valueA: any;
      let valueB: any;

      switch (this.documentSortBy) {
        case 'name':
          valueA = a.name;
          valueB = b.name;
          break;
        case 'createdAt':
          valueA = new Date(a.createdAt).getTime();
          valueB = new Date(b.createdAt).getTime();
          break;
        case 'size':
          valueA = a.metadata?.size || 0;
          valueB = b.metadata?.size || 0;
          break;
        default:
          valueA = a.name;
          valueB = b.name;
      }

      // Handle string comparisons
      if (typeof valueA === 'string' && typeof valueB === 'string') {
        const compareResult = valueA.localeCompare(valueB);
        return this.documentSortDirection === 'asc' ? compareResult : -compareResult;
      }

      // Handle number comparisons
      const compareResult = valueA - valueB;
      return this.documentSortDirection === 'asc' ? compareResult : -compareResult;
    });

    this.filteredDocuments = result;
    this.updatePagination();
  }

  /**
   * Toggle sort direction
   */
  toggleSortDirection(): void {
    this.documentSortDirection = this.documentSortDirection === 'asc' ? 'desc' : 'asc';
    this.filterDocuments();
  }

  /**
   * Reset document filters
   */
  resetDocumentFilters(): void {
    this.documentSearchQuery = '';
    this.documentFormatFilter = '';
    this.documentSortBy = 'name';
    this.documentSortDirection = 'asc';
    this.filterDocuments();
  }

  /**
   * Update pagination after filtering
   */
  updatePagination(): void {
    this.totalDocumentsPages = Math.max(1, Math.ceil(this.filteredDocuments.length / this.itemsPerPage));

    if (this.currentDocumentsPage > this.totalDocumentsPages) {
      this.currentDocumentsPage = 1;
    }
  }

  /**
   * Get currently visible documents based on pagination
   */
  get paginatedDocuments(): Document[] {
    const startIndex = (this.currentDocumentsPage - 1) * this.itemsPerPage;
    const endIndex = Math.min(startIndex + this.itemsPerPage, this.filteredDocuments.length);

    return this.filteredDocuments.slice(startIndex, endIndex);
  }

  /**
   * Get pagination display info
   */
  get paginationStart(): number {
    if (this.filteredDocuments.length === 0) return 0;
    return (this.currentDocumentsPage - 1) * this.itemsPerPage + 1;
  }

  get paginationEnd(): number {
    return Math.min(
      this.currentDocumentsPage * this.itemsPerPage,
      this.filteredDocuments.length
    );
  }

  /**
   * Navigate to specific page
   */
  goToDocumentsPage(page: number): void {
    if (page < 1 || page > this.totalDocumentsPages) return;
    this.currentDocumentsPage = page;
  }

  /**
   * Get array of page numbers to display
   */
  getDocumentPageNumbers(): number[] {
    const totalPages = this.totalDocumentsPages;
    const currentPage = this.currentDocumentsPage;
    const maxVisiblePages = 5;

    if (totalPages <= maxVisiblePages) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }

    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = startPage + maxVisiblePages - 1;

    if (endPage > totalPages) {
      endPage = totalPages;
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    return Array.from({ length: endPage - startPage + 1 }, (_, i) => startPage + i);
  }

  /**
   * Preview a document
   */
  previewDocument(document: Document): void {
    this.currentPreviewDocument = document;
    this.previewingDocument = true;
    this.isLoadingPreview = true;
    this.documentPreviewContent = '';
    this.documentPreviewError = null;
    this.documentPreviewHeaders = [];
    this.documentPreviewData = [];

    const format = this.getDocumentFormat(document);

    // For a real implementation, we would call an API to get the document content
    // This is a simplified simulation for demonstration
    setTimeout(() => {
      this.isLoadingPreview = false;

      // Simulate preview based on format
      if (format === 'CSV') {
        this.documentPreviewHeaders = ['ID', 'Query', 'Category', 'Response'];
        this.documentPreviewData = [
          { ID: '001', Query: 'How do I reset my password?', Category: 'Account', Response: 'You can reset your password by...' },
          { ID: '002', Query: 'Where can I find billing information?', Category: 'Billing', Response: 'Your billing information is available in...' },
          { ID: '003', Query: 'Product doesn\'t start after update', Category: 'Technical', Response: 'Please try the following troubleshooting steps...' }
        ];
      } else if (format === 'TXT') {
        this.documentPreviewContent = document.content || 'No content available for preview.';
      } else if (format === 'JSON') {
        try {
          // Try to pretty-print JSON if content is available
          const jsonContent = document.content || '{"message": "Sample JSON data for preview."}';
          this.documentPreviewContent = JSON.stringify(JSON.parse(jsonContent), null, 2);
        } catch (e) {
          this.documentPreviewContent = '{"error": "Invalid JSON content"}';
        }
      } else {
        this.documentPreviewError = `Preview not available for ${format} files`;
      }
    }, 1000);
  }

  /**
   * Format JSON for display
   */
  formatJson(jsonString: string): string {
    try {
      return JSON.stringify(JSON.parse(jsonString), null, 2);
    } catch (e) {
      return jsonString;
    }
  }

  /**
   * Close document preview modal
   */
  closeDocumentPreview(): void {
    this.previewingDocument = false;
    this.currentPreviewDocument = null;
  }

  /**
   * Download document
   */
  downloadDocument(): void {
    if (!this.currentPreviewDocument) return;

    // For demo purposes, we'll just show an alert
    // In a real application, this would initiate a file download
    this.alertService.showAlert({
      show: true,
      message: `Download started for ${this.currentPreviewDocument.name}`,
      title: 'Info'
    });
  }

  /**
   * Confirm and delete document
   */
  confirmDeleteDocument(event: Event, documentId: string): void {
    event.stopPropagation(); // Prevent triggering other click events

    if (confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
      // Simulated delete - In a real application, this would call an API
      this.documents = this.documents.filter(doc => doc.id !== documentId);
      this.filterDocuments();

      this.alertService.showAlert({
        show: true,
        message: 'Document deleted successfully',
        title: 'Success'
      });
    }
  }

  /**
   * Go back to datasets list
   */
  goBack(event: Event): void {
    event.preventDefault();
    this.router.navigate(['app/datasets/datasets']);
  }

  /**
   * Format date for display
   */
  formatDate(dateString: string | undefined): string {
    if (!dateString) return 'N/A';
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

  /**
   * Format file size for display
   */
  formatFileSize(bytes: number | undefined): string {
    if (bytes === undefined || bytes === 0) return 'N/A';

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }

    return `${value.toFixed(1)} ${units[unitIndex]}`;
  }

  /**
   * Get status text for display
   */
  get statusText(): string {
    if (!this.dataset) return '';

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

  /**
   * Get status CSS class
   */
  get statusClass(): string {
    if (!this.dataset) return '';

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

  /**
   * Get formatted file size
   */
  get formattedSize(): string {
    if (!this.dataset || !this.dataset.size) return 'N/A';
    return this.formatFileSize(this.dataset.size);
  }
}