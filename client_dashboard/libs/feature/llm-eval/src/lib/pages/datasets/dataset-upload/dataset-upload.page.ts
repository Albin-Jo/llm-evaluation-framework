/* Path: libs/feature/llm-eval/src/lib/pages/datasets/dataset-upload/dataset-upload.page.ts */
import { Component, OnDestroy, OnInit, ViewChild, ElementRef, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, ActivatedRoute } from '@angular/router';
import { Subject, takeUntil, finalize } from 'rxjs';
import { DatasetUploadRequest } from '@ngtx-apps/data-access/models';
import { DatasetService } from '@ngtx-apps/data-access/services';
import { AlertService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-dataset-upload',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './dataset-upload.page.html',
  styleUrls: ['./dataset-upload.page.scss']
})
export class DatasetUploadPage implements OnInit, OnDestroy {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  // Form
  uploadForm: FormGroup;

  // Upload state
  isUploading = false;
  isDragging = false;
  uploadProgress = 0;
  selectedFiles: File[] = [];
  selectedTags: string[] = [];
  existingDatasetId: string | null = null;
  uploadCompleted = false;
  uploadError = false;
  errorMessage: string | null = null;
  createdDatasetId: string | null = null;

  // Validation
  validFileTypes = [
    'application/pdf',
    'text/plain',
    'text/csv',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // docx
    'application/json'
  ];
  validFileExtensions = ['.pdf', '.txt', '.csv', '.docx', '.json'];
  maxFileSize = 50 * 1024 * 1024; // 50MB

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
  private pendingUploadRequest: DatasetUploadRequest | null = null;

  constructor(
    private fb: FormBuilder,
    private datasetService: DatasetService,
    private alertService: AlertService,
    private router: Router,
    private route: ActivatedRoute
  ) {
    this.uploadForm = this.fb.group({
      name: ['', [Validators.required, Validators.maxLength(100)]],
      type: ['user_query', Validators.required],
      description: ['', Validators.maxLength(500)],
      files: [null, Validators.required]
    });
  }

  ngOnInit(): void {
    // Check if uploading to an existing dataset
    this.route.queryParams
      .pipe(takeUntil(this.destroy$))
      .subscribe(params => {
        if (params['datasetId']) {
          this.existingDatasetId = params['datasetId'];

          // If adding to existing dataset, name and description are not required
          this.uploadForm.get('name')?.clearValidators();
          this.uploadForm.get('name')?.updateValueAndValidity();

          this.uploadForm.get('type')?.clearValidators();
          this.uploadForm.get('type')?.updateValueAndValidity();
        }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Open file selector dialog
   */
  openFileSelector(): void {
    this.fileInput.nativeElement.click();
  }

  /**
   * Handle file selection
   */
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.addFiles(Array.from(input.files));
    }
  }

  /**
   * Handle drag over event
   */
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = true;
  }

  /**
   * Handle drag leave event
   */
  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;
  }

  /**
   * Handle file drop event
   */
  onFileDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;

    if (event.dataTransfer && event.dataTransfer.files.length > 0) {
      this.addFiles(Array.from(event.dataTransfer.files));
    }
  }

  /**
   * Add files to selection
   */
  addFiles(files: File[]): void {
    // Only allow one file, replace existing selection
    this.selectedFiles = files.slice(0, 1);
    this.uploadForm.patchValue({ files: this.selectedFiles });
    this.uploadForm.markAsDirty();
  }

  /**
   * Format file size in bytes to a readable format
   */
  formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  /**
   * Toggle tag selection
   */
  toggleTag(tag: string): void {
    if (this.selectedTags.includes(tag)) {
      this.selectedTags = this.selectedTags.filter(t => t !== tag);
    } else {
      this.selectedTags = [...this.selectedTags, tag];
    }
  }

  /**
   * Remove a file from the selected files
   */
  removeFile(index: number): void {
    this.selectedFiles.splice(index, 1);
    this.uploadForm.patchValue({ files: this.selectedFiles.length ? this.selectedFiles : null });
    this.uploadForm.markAsDirty();
  }

  /**
   * Check if file is valid
   */
  isValidFile(file: File): boolean {
    // Check file type
    const fileExt = file.name.toLowerCase().split('.').pop();
    const isValidType = this.validFileTypes.includes(file.type) ||
                       (fileExt ? this.validFileExtensions.some(ext => ext.endsWith(fileExt)) : false);

    // Check file size
    const isValidSize = file.size <= this.maxFileSize;

    return isValidType && isValidSize;
  }

  /**
   * Check if we have valid files
   */
  hasValidFiles(): boolean {
    return this.selectedFiles.some(file => this.isValidFile(file));
  }

  /**
   * Get file icon class based on file type
   */
  getFileIconClass(file: File): string {
    const fileType = file.type;
    const fileName = file.name.toLowerCase();

    if (fileType === 'application/pdf' || fileName.endsWith('.pdf')) {
      return 'icon-pdf';
    } else if (fileType === 'text/csv' || fileName.endsWith('.csv')) {
      return 'icon-csv';
    } else if (fileType === 'text/plain' || fileName.endsWith('.txt')) {
      return 'icon-txt';
    } else if (fileType.includes('wordprocessingml') || fileName.endsWith('.docx') || fileName.endsWith('.doc')) {
      return 'icon-docx';
    } else if (fileType === 'application/json' || fileName.endsWith('.json')) {
      return 'icon-json';
    } else {
      return 'icon-unknown';
    }
  }

  /**
   * Get file icon text based on file type
   */
  getFileIconText(file: File): string {
    const fileType = file.type;
    const fileName = file.name.toLowerCase();

    if (fileType === 'application/pdf' || fileName.endsWith('.pdf')) {
      return 'PDF';
    } else if (fileType === 'text/csv' || fileName.endsWith('.csv')) {
      return 'CSV';
    } else if (fileType === 'text/plain' || fileName.endsWith('.txt')) {
      return 'TXT';
    } else if (fileType.includes('wordprocessingml') || fileName.endsWith('.docx') || fileName.endsWith('.doc')) {
      return 'DOC';
    } else if (fileType === 'application/json' || fileName.endsWith('.json')) {
      return 'JSON';
    } else {
      const ext = fileName.split('.').pop()?.toUpperCase();
      return ext || '?';
    }
  }

  /**
   * Submit form and upload dataset
   */
  onSubmit(): void {
    // Skip validation if we're adding to an existing dataset
    if (this.existingDatasetId) {
      if (this.selectedFiles.length === 0 || this.isUploading) {
        return;
      }
    } else {
      if (this.uploadForm.invalid || this.isUploading || this.selectedFiles.length === 0) {
        this.markFormGroupTouched(this.uploadForm);
        return;
      }
    }

    this.isUploading = true;
    this.uploadProgress = 0;
    this.uploadError = false;
    this.errorMessage = null;

    // Set up progress simulation
    const progressInterval = setInterval(() => {
      if (this.uploadProgress < 90) {
        this.uploadProgress += Math.floor(Math.random() * 5) + 1;
      }
    }, 200);

    // If existing dataset ID, upload to that dataset
    if (this.existingDatasetId) {
      this.datasetService.uploadDocumentsToDataset(this.existingDatasetId, this.selectedFiles)
        .pipe(
          takeUntil(this.destroy$),
          finalize(() => clearInterval(progressInterval))
        )
        .subscribe({
          next: (response) => {
            this.uploadProgress = 100;
            this.handleUploadSuccess();
          },
          error: (error) => {
            this.handleUploadError(error);
            clearInterval(progressInterval);
          }
        });
    } else {
      // Create a new dataset
      const formValues = this.uploadForm.value;
      const uploadRequest: DatasetUploadRequest = {
        name: formValues.name,
        description: formValues.description || '',
        tags: this.selectedTags,
        files: this.selectedFiles
      };

      this.pendingUploadRequest = uploadRequest;
      this.datasetService.uploadDataset(uploadRequest)
        .pipe(
          takeUntil(this.destroy$),
          finalize(() => clearInterval(progressInterval))
        )
        .subscribe({
          next: (response) => {
            this.uploadProgress = 100;
            this.createdDatasetId = response.id;
            this.handleUploadSuccess();
          },
          error: (error) => {
            this.handleUploadError(error);
            clearInterval(progressInterval);
          }
        });
    }
  }

  /**
   * Handle successful upload
   */
  private handleUploadSuccess(): void {
    // Show success message after a short delay to ensure progress reaches 100%
    setTimeout(() => {
      this.isUploading = false;
      this.uploadCompleted = true;
      this.pendingUploadRequest = null;
    }, 500);
  }

  /**
   * Handle upload error
   */
  private handleUploadError(error: any): void {
    console.error('Error uploading:', error);
    this.isUploading = false;
    this.uploadError = true;

    if (error?.error?.detail) {
      this.errorMessage = typeof error.error.detail === 'string'
        ? error.error.detail
        : 'Failed to upload. Please try again.';
    } else if (error?.message) {
      this.errorMessage = error.message;
    } else {
      this.errorMessage = 'Failed to upload dataset. Please try again.';
    }
  }

  /**
   * Retry a failed upload
   */
  retryUpload(): void {
    this.uploadError = false;
    this.errorMessage = null;

    if (this.pendingUploadRequest) {
      // Re-submit the form with the same values
      this.onSubmit();
    } else {
      // Just close the error dialog and let user try again manually
      this.dismissError();
    }
  }

  /**
   * Dismiss error dialog
   */
  dismissError(): void {
    this.uploadError = false;
    this.errorMessage = null;
  }

  /**
   * Helper method to mark all form controls as touched
   */
  private markFormGroupTouched(formGroup: FormGroup) {
    Object.values(formGroup.controls).forEach(control => {
      control.markAsTouched();
      if ((control as any).controls) {
        this.markFormGroupTouched(control as FormGroup);
      }
    });
  }

  /**
   * Cancel upload and navigate back
   */
  cancel(event: Event): void {
    event.preventDefault();

    if (this.isUploading) {
      if (confirm('Uploading is in progress. Are you sure you want to cancel?')) {
        // In a real application, we would cancel the upload request here
        this.navigateBack();
      }
    } else {
      this.navigateBack();
    }
  }

  /**
   * Navigate back based on context
   */
  private navigateBack(): void {
    if (this.existingDatasetId) {
      this.router.navigate(['/app/datasets/datasets', this.existingDatasetId]);
    } else {
      this.router.navigate(['/app/datasets/datasets']);
    }
  }

  /**
   * View dataset after successful upload
   */
  viewDataset(): void {
    if (this.existingDatasetId) {
      this.router.navigate(['/app/datasets/datasets', this.existingDatasetId]);
    } else if (this.createdDatasetId) {
      this.router.navigate(['/app/datasets/datasets', this.createdDatasetId]);
    } else {
      this.router.navigate(['/app/datasets/datasets']);
    }
  }

  /**
   * Form control getters
   */
  get nameControl() { return this.uploadForm.get('name'); }
  get descriptionControl() { return this.uploadForm.get('description'); }
  get filesControl() { return this.uploadForm.get('files'); }