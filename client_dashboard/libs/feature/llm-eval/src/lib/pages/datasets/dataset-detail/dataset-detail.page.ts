/* Path: libs/feature/llm-eval/src/lib/pages/datasets/dataset-detail/dataset-detail.page.ts */
import { Component, OnDestroy, OnInit, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Subject, takeUntil, switchMap, of, catchError } from 'rxjs';
import {
  Dataset,
  DatasetStatus,
  DatasetUpdateRequest,
  Document
} from '@ngtx-apps/data-access/models';
import { DatasetService } from '@ngtx-apps/data-access/services';
import { AlertService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-dataset-detail',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './dataset-detail.page.html',
  styleUrls: ['./dataset-detail.page.scss']
})
export class DatasetDetailPage implements OnInit, OnDestroy {
  dataset: Dataset | null = null;
  documents: Document[] = [];
  isLoading = true;
  isSaving = false;
  errorMessage: string | null = null;
  selectedTags: string[] = [];

  editForm: FormGroup;

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

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private datasetService: DatasetService,
    private alertService: AlertService,
    private fb: FormBuilder
  ) {
    this.editForm = this.fb.group({
      name: ['', [Validators.required, Validators.maxLength(100)]],
      description: ['', Validators.maxLength(500)]
    });
  }

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
            this.selectedTags = this.dataset.tags || [];
            this.initializeForm();
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
   * Initialize the form with dataset values
   */
  initializeForm(): void {
    if (!this.dataset) return;

    this.editForm.patchValue({
      name: this.dataset.name,
      description: this.dataset.description || ''
    });
  }

  /**
   * Toggle a tag selection
   */
  toggleTag(tag: string): void {
    if (this.selectedTags.includes(tag)) {
      this.selectedTags = this.selectedTags.filter(t => t !== tag);
    } else {
      this.selectedTags = [...this.selectedTags, tag];
    }
  }

  /**
   * Save dataset changes
   */
  saveChanges(event: Event): void {
    event.preventDefault();
    if (this.editForm.invalid || !this.dataset) return;

    this.isSaving = true;

    const formValues = this.editForm.value;
    const updateRequest: DatasetUpdateRequest = {
      name: formValues.name,
      description: formValues.description,
      tags: this.selectedTags
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
          // Navigate back to the dataset list
          this.router.navigate(['app/datasets/datasets']);
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
   * Form getters for easy access in template
   */
  get nameControl() { return this.editForm.get('name'); }
  get descriptionControl() { return this.editForm.get('description'); }

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
   * Format file size for display
   */
  get formattedSize(): string {
    if (!this.dataset || !this.dataset.size) return 'N/A';

    const bytes = this.dataset.size;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  /**
   * Format creation date
   */
  get formattedDate(): string {
    if (!this.dataset || !this.dataset.createdAt) return 'N/A';
    return this.formatDate(this.dataset.createdAt);
  }
}
