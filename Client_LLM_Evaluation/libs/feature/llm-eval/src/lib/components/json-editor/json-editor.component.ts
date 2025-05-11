/* Path: libs/feature/llm-eval/src/lib/components/json-editor/json-editor.component.ts */
import { 
  Component, 
  Input, 
  Output, 
  EventEmitter, 
  OnInit, 
  OnDestroy,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  ElementRef,
  ViewChild
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Subject, debounceTime, takeUntil } from 'rxjs';
import { QracButtonComponent } from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-json-editor',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracButtonComponent
  ],
  templateUrl: './json-editor.component.html',
  styleUrls: ['./json-editor.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class JsonEditorComponent implements OnInit, OnDestroy {
  @Input() jsonValue: string = '{}';
  @Input() title: string = 'Edit JSON';
  @Input() placeholder: string = 'Enter JSON content';
  @Input() readOnly: boolean = false;
  @Input() rows: number = 6;
  @Input() maxLength: number = 10000;
  @Input() required: boolean = false;
  @Input() label: string = '';
  
  // Two-way binding for modal state
  @Input() isOpen: boolean = false;
  @Output() isOpenChange = new EventEmitter<boolean>();
  
  // Events
  @Output() valueChange = new EventEmitter<string>();
  @Output() validChange = new EventEmitter<boolean>();
  
  // Add ViewChild for the textarea to focus when modal opens
  @ViewChild('jsonTextarea') jsonTextarea?: ElementRef<HTMLTextAreaElement>;
  
  jsonControl = new FormControl('');
  isValid = true;
  previewSummary: string = '';
  errorMessage: string = '';
  
  private destroy$ = new Subject<void>();

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    // Initialize control with input value
    this.updateJsonControl(this.jsonValue);
    
    // Set up listener for changes
    this.jsonControl.valueChanges
      .pipe(
        debounceTime(300),
        takeUntil(this.destroy$)
      )
      .subscribe(value => {
        this.validateJson(value || '{}');
      });
  }
  
  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
  
  /**
   * Update the form control with new JSON value
   */
  private updateJsonControl(value: string): void {
    try {
      // Ensure the value is a string
      const jsonString = typeof value === 'object' 
        ? JSON.stringify(value, null, 2) 
        : (value || '{}');
        
      this.jsonControl.setValue(jsonString, { emitEvent: false });
      this.validateJson(jsonString);
    } catch (e) {
      this.isValid = false;
      this.errorMessage = 'Invalid JSON format';
      this.validChange.emit(false);
    }
  }
  
  /**
   * Validate JSON and update preview
   */
  private validateJson(value: string): void {
    if (!value || value.trim() === '') {
      this.previewSummary = '{}';
      this.isValid = true;
      this.errorMessage = '';
      this.validChange.emit(true);
      return;
    }
    
    try {
      const parsed = JSON.parse(value);
      this.isValid = true;
      this.errorMessage = '';
      this.validChange.emit(true);
      
      // Create preview summary
      this.generatePreviewSummary(parsed);
    } catch (e) {
      this.isValid = false;
      this.errorMessage = 'Invalid JSON format';
      this.validChange.emit(false);
    }
    
    this.cdr.markForCheck();
  }
  
  /**
   * Generate a preview of the JSON content
   */
  private generatePreviewSummary(json: any): void {
    try {
      const compact = JSON.stringify(json);
      if (compact === '{}') {
        this.previewSummary = '{}';
        return;
      }
      
      if (compact.length > 50) {
        this.previewSummary = compact.substring(0, 47) + '...';
      } else {
        this.previewSummary = compact;
      }
    } catch (e) {
      this.previewSummary = 'Error generating preview';
    }
  }
  
  /**
   * Open the editor modal
   */
  openModal(): void {
    this.isOpen = true;
    this.isOpenChange.emit(true);
    this.cdr.markForCheck();
    
    // Focus the textarea after the modal is rendered
    setTimeout(() => {
      if (this.jsonTextarea) {
        this.jsonTextarea.nativeElement.focus();
      }
    }, 100);
  }
  
  /**
   * Close the editor modal
   */
  closeModal(event?: MouseEvent): void {
    // If this is a direct call with no event, or the click was on the modal overlay itself
    // and not on its children, close the modal
    if (!event || event.target === event.currentTarget) {
      this.isOpen = false;
      this.isOpenChange.emit(false);
      this.cdr.markForCheck();
    }
  }
  
  /**
   * Prevent event propagation when clicking inside the modal
   */
  stopPropagation(event: MouseEvent): void {
    event.stopPropagation();
  }
  
  /**
   * Apply changes and close modal
   */
  applyChanges(): void {
    if (!this.isValid) {
      return;
    }
    
    const value = this.jsonControl.value || '{}';
    this.valueChange.emit(value);
    this.closeModal();
  }
  
  /**
   * Format the JSON for better readability
   */
  formatJson(): void {
    const value = this.jsonControl.value || '{}';
    
    try {
      const formatted = JSON.stringify(JSON.parse(value), null, 2);
      this.jsonControl.setValue(formatted);
      this.isValid = true;
      this.errorMessage = '';
      this.validChange.emit(true);
      this.cdr.markForCheck();
    } catch (e) {
      // Keep current value if parsing fails
      this.isValid = false;
      this.errorMessage = 'Invalid JSON format';
      this.validChange.emit(false);
    }
  }
  
  /**
   * Clear the JSON editor
   */
  clearJson(): void {
    this.jsonControl.setValue('{}');
    this.isValid = true;
    this.errorMessage = '';
    this.validChange.emit(true);
    this.cdr.markForCheck();
  }
  
  /**
   * Add a new property to the JSON object
   */
  addNewProperty(): void {
    try {
      const currentJson = JSON.parse(this.jsonControl.value || '{}');
      // Generate a unique key
      let newKey = 'newProperty';
      let counter = 1;
      
      while (currentJson.hasOwnProperty(newKey)) {
        newKey = `newProperty${counter}`;
        counter++;
      }
      
      currentJson[newKey] = '';
      this.jsonControl.setValue(JSON.stringify(currentJson, null, 2));
      this.isValid = true;
      this.errorMessage = '';
      this.validChange.emit(true);
      this.cdr.markForCheck();
    } catch (e) {
      // Keep current value if parsing fails
      this.isValid = false;
      this.errorMessage = 'Invalid JSON format';
      this.validChange.emit(false);
    }
  }
}