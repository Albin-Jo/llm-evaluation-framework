/* Path: libs/feature/llm-eval/src/lib/pages/datasets/dataset-upload/dataset-upload.page.ts */
import { Component, OnDestroy, OnInit, ViewChild, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, ActivatedRoute } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { DatasetUploadRequest } from '@ngtx-apps/data-access/models';
import { DatasetService } from '@ngtx-apps/data-access/services';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracTextAreaComponent,
  QracuploadComponent,
  QracTagButtonComponent
} from '@ngtx-apps/ui/components';
import { AlertService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-dataset-upload',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    QracTextBoxComponent,
    QracTextAreaComponent,
    QracuploadComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './dataset-upload.page.html',
  styleUrls: ['./dataset-upload.page.scss']
})
export class DatasetUploadPage implements OnInit, OnDestroy {
  @ViewChild(QracuploadComponent) uploadComponent?: QracuploadComponent;
  uploadForm: FormGroup;
  isUploading = false;
  selectedFiles: File[] = [];
  selectedTags: string[] = [];
  existingDatasetId: string | null = null;

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
    private fb: FormBuilder,
    private datasetService: DatasetService,
    private alertService: AlertService,
    private router: Router,
    private route: ActivatedRoute
  ) {
    this.uploadForm = this.fb.group({
      name: ['', [Validators.required, Validators.maxLength(100)]],
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
        }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Formats file size in bytes to a readable format
   */
  formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  /**
   * Handles file selection from various inputs
   */
  onFileSelected(event: any): void {
    if (event && event.filesData && event.filesData.length > 0) {
      // Handle Syncfusion event
      const files: File[] = [];

      for (const fileInfo of event.filesData) {
        if (fileInfo.rawFile instanceof File) {
          files.push(fileInfo.rawFile);
        }
      }

      this.selectedFiles = files;

      if (files.length > 0) {
        this.uploadForm.patchValue({ files: files });
        this.uploadForm.markAsDirty();
      }
    } else if (event && event.target && event.target.files) {
      // Handle standard file input event
      this.selectedFiles = Array.from(event.target.files);
      this.uploadForm.patchValue({ files: this.selectedFiles });
      this.uploadForm.markAsDirty();
    }
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

    // If existing dataset ID, upload to that dataset
    if (this.existingDatasetId) {
      this.uploadToExistingDataset();
    } else {
      // Create a new dataset
      const formValues = this.uploadForm.value;
      const uploadRequest: DatasetUploadRequest = {
        name: formValues.name,
        description: formValues.description || '',
        tags: this.selectedTags,
        files: this.selectedFiles
      };
      this.createNewDataset(uploadRequest);
    }
  }

  /**
   * Create a new dataset with uploaded files
   */
  private createNewDataset(uploadRequest: DatasetUploadRequest): void {
    this.datasetService.uploadDataset(uploadRequest)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.alertService.showAlert({
            show: true,
            message: 'Dataset uploaded successfully',
            title: 'Success'
          });
          this.router.navigate(['app/datasets/datasets']);
        },
        error: (error) => {
          this.alertService.showAlert({
            show: true,
            message: 'Failed to upload dataset. Please try again.',
            title: 'Error'
          });
          this.isUploading = false;
          console.error('Error uploading dataset:', error);
        }
      });
  }

  /**
   * Upload files to an existing dataset
   */
  private uploadToExistingDataset(): void {
    if (!this.existingDatasetId) return;

    this.datasetService.uploadDocumentsToDataset(this.existingDatasetId, this.selectedFiles)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.alertService.showAlert({
            show: true,
            message: 'Documents uploaded successfully',
            title: 'Success'
          });
          this.router.navigate(['app/datasets/datasets', this.existingDatasetId]);
        },
        error: (error) => {
          this.alertService.showAlert({
            show: true,
            message: 'Failed to upload documents. Please try again.',
            title: 'Error'
          });
          this.isUploading = false;
          console.error('Error uploading documents:', error);
        }
      });
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
    if (this.existingDatasetId) {
      this.router.navigate(['/app/datasets/datasets', this.existingDatasetId]);
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
}
