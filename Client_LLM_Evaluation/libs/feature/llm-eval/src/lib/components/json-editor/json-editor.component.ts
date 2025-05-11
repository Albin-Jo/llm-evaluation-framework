/* Path: libs/feature/llm-eval/src/lib/components/json-editor/json-editor.component.ts */
import {
  Component,
  Input,
  Output,
  EventEmitter,
  OnInit,
  OnChanges,
  SimpleChanges,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  NO_ERRORS_SCHEMA
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, FormArray, ReactiveFormsModule, Validators } from '@angular/forms';

interface KeyValuePair {
  key: string;
  value: any;
}

@Component({
  selector: 'app-json-editor',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './json-editor.component.html',
  styleUrls: ['./json-editor.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class JsonEditorComponent implements OnInit, OnChanges {
  @Input() jsonValue: string = '{}';
  @Input() title: string = 'Edit JSON';
  @Input() isOpen: boolean = false;
  @Output() isOpenChange = new EventEmitter<boolean>();
  @Output() valueChange = new EventEmitter<string>();
  @Output() validChange = new EventEmitter<boolean>();

  jsonForm!: FormGroup;
  isRawMode: boolean = false;
  rawJsonText: string = '{}';
  jsonError: string | null = null;
  keyValuePairs: KeyValuePair[] = [];
  isValid: boolean = true;
  formattedJsonString: string = '';

  constructor(
    private fb: FormBuilder,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.initForm();
    this.parseJson();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['jsonValue'] && !changes['jsonValue'].firstChange) {
      this.parseJson();
    }
  }

  private initForm(): void {
    this.jsonForm = this.fb.group({
      keyValuePairs: this.fb.array([]),
      rawJson: ['{}', this.validateJson]
    });
  }

  parseJson(): void {
    try {
      // Handle undefined, null, or empty string input
      if (!this.jsonValue || this.jsonValue === '' || this.jsonValue === 'undefined' || this.jsonValue === 'null') {
        this.jsonValue = '{}';
      }

      // If it's already a JSON object, stringify it
      if (typeof this.jsonValue === 'object') {
        this.jsonValue = JSON.stringify(this.jsonValue);
      }

      // Parse the JSON string
      const jsonObj = JSON.parse(this.jsonValue);
      this.formattedJsonString = JSON.stringify(jsonObj, null, 2);

      // Convert to key-value pairs
      this.keyValuePairs = Object.entries(jsonObj).map(([key, value]) => ({
        key,
        value: typeof value === 'object' ? JSON.stringify(value) : value
      }));

      this.rawJsonText = this.formattedJsonString;
      this.updateFormArray();
      this.jsonForm.get('rawJson')?.setValue(this.formattedJsonString);
      this.jsonError = null;
      this.isValid = true;
      this.validChange.emit(true);
    } catch (error) {
      console.error('Invalid JSON input:', error);
      this.jsonError = 'Invalid JSON format';
      this.keyValuePairs = [];
      this.rawJsonText = this.jsonValue || '{}';
      this.updateFormArray();
      this.isValid = false;
      this.validChange.emit(false);
    }

    this.cdr.markForCheck();
  }

  updateFormArray(): void {
    const formArray = this.jsonForm.get('keyValuePairs') as FormArray;

    // Clear existing controls
    while (formArray.length) {
      formArray.removeAt(0);
    }

    // Add new controls
    this.keyValuePairs.forEach(pair => {
      formArray.push(
        this.fb.group({
          key: [pair.key, [Validators.required]],
          value: [pair.value, []]
        })
      );
    });
  }

  get keyValuePairsFormArray(): FormArray {
    return this.jsonForm.get('keyValuePairs') as FormArray;
  }

  addKeyValuePair(): void {
    const formArray = this.jsonForm.get('keyValuePairs') as FormArray;
    formArray.push(
      this.fb.group({
        key: ['', [Validators.required]],
        value: ['', []]
      })
    );
    this.cdr.markForCheck();
  }

  removeKeyValuePair(index: number): void {
    const formArray = this.jsonForm.get('keyValuePairs') as FormArray;
    formArray.removeAt(index);
    this.cdr.markForCheck();
  }

  toggleMode(): void {
    if (this.isRawMode) {
      // Switching from raw to structured - parse the raw JSON
      try {
        const rawValue = this.jsonForm.get('rawJson')?.value;
        if (rawValue) {
          const jsonObj = JSON.parse(rawValue);
          this.keyValuePairs = Object.entries(jsonObj).map(([key, value]) => ({
            key,
            value: typeof value === 'object' ? JSON.stringify(value) : value
          }));
          this.updateFormArray();
          this.jsonError = null;
          this.isValid = true;
          this.validChange.emit(true);
        }
      } catch (error) {
        this.jsonError = 'Invalid JSON format';
        this.isValid = false;
        this.validChange.emit(false);
        // Stay in raw mode if parsing fails
        this.cdr.markForCheck();
        return;
      }
    } else {
      // Switching from structured to raw - build JSON from key-value pairs
      try {
        const jsonObj = this.buildJsonFromForm();
        this.jsonForm.get('rawJson')?.setValue(JSON.stringify(jsonObj, null, 2));
        this.jsonError = null;
        this.isValid = true;
        this.validChange.emit(true);
      } catch (error) {
        this.jsonError = 'Error converting to JSON';
        this.isValid = false;
        this.validChange.emit(false);
        this.cdr.markForCheck();
        return;
      }
    }

    this.isRawMode = !this.isRawMode;
    this.cdr.markForCheck();
  }

  buildJsonFromForm(): any {
    const result: any = {};
    const formArray = this.jsonForm.get('keyValuePairs') as FormArray;

    formArray.controls.forEach(control => {
      const key = control.get('key')?.value;
      let value = control.get('value')?.value;

      if (key) {
        // Try to parse values that look like objects or arrays
        if (typeof value === 'string') {
          if ((value.startsWith('{') && value.endsWith('}')) ||
              (value.startsWith('[') && value.endsWith(']'))) {
            try {
              value = JSON.parse(value);
            } catch (e) {
              // Keep as string if parsing fails
            }
          } else if (value === 'true' || value === 'false') {
            value = value === 'true';
          } else if (!isNaN(Number(value)) && value !== '') {
            value = Number(value);
          }
        }

        result[key] = value;
      }
    });

    return result;
  }

  validateJson(control: any): {[key: string]: any} | null {
    try {
      if (!control.value || control.value === '{}') {
        return null;
      }
      JSON.parse(control.value);
      return null;
    } catch (error) {
      return { invalidJson: true };
    }
  }

  save(): void {
    if (this.isRawMode) {
      // Validate and save raw JSON
      const rawValue = this.jsonForm.get('rawJson')?.value;
      try {
        if (rawValue) {
          const parsed = JSON.parse(rawValue);
          const jsonStr = JSON.stringify(parsed);
          this.valueChange.emit(jsonStr);
          this.close();
          this.isValid = true;
          this.validChange.emit(true);
        }
      } catch (error) {
        this.jsonError = 'Invalid JSON format';
        this.isValid = false;
        this.validChange.emit(false);
        this.cdr.markForCheck();
      }
    } else {
      // Save from structured editor
      try {
        const jsonObj = this.buildJsonFromForm();
        const jsonStr = JSON.stringify(jsonObj);
        this.valueChange.emit(jsonStr);
        this.formattedJsonString = JSON.stringify(jsonObj, null, 2);
        this.close();
        this.isValid = true;
        this.validChange.emit(true);
      } catch (error) {
        this.jsonError = 'Error saving JSON';
        this.isValid = false;
        this.validChange.emit(false);
        this.cdr.markForCheck();
      }
    }
  }

  open(): void {
    this.isOpen = true;
    this.parseJson();
    this.isOpenChange.emit(true);
    this.cdr.markForCheck();
  }

  close(): void {
    this.isOpen = false;
    this.isOpenChange.emit(false);
    this.cdr.markForCheck();
  }

  // Get formatted preview for display in main form
  getPreviewText(): string {
    try {
      if (!this.jsonValue || this.jsonValue === '{}') {
        return 'No data configured';
      }

      const jsonObj = typeof this.jsonValue === 'string' ?
        JSON.parse(this.jsonValue) : this.jsonValue;

      // Count the number of keys
      const keyCount = Object.keys(jsonObj).length;

      if (keyCount === 0) {
        return 'Empty object';
      }

      // Create a brief summary
      return JSON.stringify(jsonObj);
    } catch (e) {
      return 'Invalid JSON';
    }
  }
}
