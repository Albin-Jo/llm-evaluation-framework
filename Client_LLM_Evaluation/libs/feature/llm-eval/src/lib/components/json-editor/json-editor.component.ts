/* Path: libs/feature/llm-eval/src/lib/components/simple-json-editor/simple-json-editor.component.ts */
import {
  Component,
  Input,
  Output,
  EventEmitter,
  OnInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { QracButtonComponent } from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-simple-json-editor',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracButtonComponent
  ],
  template: `
    <div class="simple-json-editor">
      <!-- Label -->
      <div *ngIf="label" class="editor-label" [class.required]="required">{{ label }}</div>

      <!-- JSON Preview Container -->
      <div class="json-preview-container" [class.invalid]="!isValid" (click)="showEditor = !showEditor">
        <div class="json-preview">{{ previewText }}</div>
        <div class="preview-actions">
          <span *ngIf="!isValid" class="error-indicator">⚠️</span>
          <span class="edit-icon">✎</span>
        </div>
      </div>

      <!-- Error Message -->
      <div *ngIf="!isValid" class="json-error">{{ errorMessage }}</div>

      <!-- Expanded Editor -->
      <div *ngIf="showEditor" class="editor-container">
        <textarea
          class="json-textarea"
          [formControl]="jsonControl"
          placeholder="Enter JSON..."
          rows="5">
        </textarea>

        <div class="editor-actions">
          <qrac-button
            [label]="'Format'"
            [type]="'secondary'"
            size="small"
            (buttonClick)="formatJson()">
          </qrac-button>

          <qrac-button
            [label]="'Apply'"
            [type]="'primary'"
            [disabled]="!isValid"
            size="small"
            (buttonClick)="applyChanges()">
          </qrac-button>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .simple-json-editor {
      width: 100%;
      margin-bottom: 8px;
    }

    .editor-label {
      font-size: 14px;
      font-weight: 500;
      margin-bottom: 4px;
    }

    .editor-label.required::after {
      content: "*";
      color: #dc3545;
      margin-left: 4px;
    }

    .json-preview-container {
      width: 100%;
      padding: 8px 12px;
      border: 1px solid #ced4da;
      border-radius: 4px;
      background-color: #f8f9fa;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: border-color 0.2s, box-shadow 0.2s;
    }

    .json-preview-container:hover {
      border-color: #80bdff;
      background-color: #f0f7ff;
    }

    .json-preview-container.invalid {
      border-color: #dc3545;
      background-color: #fff5f5;
    }

    .json-preview {
      font-family: monospace;
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex: 1;
    }

    .preview-actions {
      display: flex;
      gap: 8px;
    }

    .error-indicator {
      color: #dc3545;
    }

    .edit-icon {
      color: #6c757d;
    }

    .json-error {
      font-size: 12px;
      color: #dc3545;
      margin-top: 4px;
    }

    .editor-container {
      margin-top: 8px;
      border: 1px solid #ced4da;
      border-radius: 4px;
      padding: 8px;
      background-color: #fff;
    }

    .json-textarea {
      width: 100%;
      padding: 8px;
      border: 1px solid #ced4da;
      border-radius: 4px;
      font-family: monospace;
      font-size: 13px;
      resize: vertical;
    }

    .editor-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 8px;
    }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class SimpleJsonEditorComponent implements OnInit {
  @Input() value: string = '{}';
  @Input() label: string = '';
  @Input() required: boolean = false;

  @Output() valueChange = new EventEmitter<string>();
  @Output() validChange = new EventEmitter<boolean>();

  jsonControl = new FormControl('{}');
  isValid = true;
  errorMessage = '';
  showEditor = false;
  previewText = '{}';

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    // Initialize with input value
    this.updateValue(this.value);

    // Listen for input changes
    this.jsonControl.valueChanges.subscribe(value => {
      this.validateJson(value || '{}');
    });
  }

  updateValue(value: string): void {
    try {
      // Handle object values passed in
      const jsonString = typeof value === 'object'
        ? JSON.stringify(value, null, 2)
        : (value || '{}');

      this.jsonControl.setValue(jsonString);
      this.validateJson(jsonString);
    } catch (e) {
      this.jsonControl.setValue('{}');
      this.validateJson('{}');
    }
  }

  validateJson(value: string): void {
    if (!value || value.trim() === '') {
      value = '{}';
    }

    try {
      const parsed = JSON.parse(value);
      this.isValid = true;
      this.errorMessage = '';
      this.validChange.emit(true);

      // Generate preview text
      this.previewText = this.generatePreview(parsed);
      this.cdr.markForCheck();
    } catch (e) {
      this.isValid = false;
      this.errorMessage = 'Invalid JSON format';
      this.validChange.emit(false);
      this.cdr.markForCheck();
    }
  }

  generatePreview(json: any): string {
    try {
      const str = JSON.stringify(json);
      if (str.length > 50) {
        return str.substring(0, 47) + '...';
      }
      return str;
    } catch (e) {
      return '{}';
    }
  }

  formatJson(): void {
    try {
      const value = this.jsonControl.value || '{}';
      const parsed = JSON.parse(value);
      const formatted = JSON.stringify(parsed, null, 2);
      this.jsonControl.setValue(formatted);
    } catch (e) {
      // If parsing fails, do nothing
    }
  }

  applyChanges(): void {
    if (!this.isValid) {
      return;
    }

    const value = this.jsonControl.value || '{}';
    this.valueChange.emit(value);
    this.showEditor = false;
    this.cdr.markForCheck();
  }
}