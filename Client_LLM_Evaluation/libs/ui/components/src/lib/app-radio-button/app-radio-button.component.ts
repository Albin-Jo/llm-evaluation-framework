import { Component, Input, forwardRef } from '@angular/core';
import { NG_VALUE_ACCESSOR, ControlValueAccessor, FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-radio-button',
  templateUrl: './app-radio-button.component.html',
  styleUrls: ['./app-radio-button.component.scss'],
  imports: [CommonModule, FormsModule],
  providers: [{
    provide: NG_VALUE_ACCESSOR,
    useExisting: forwardRef(() => RadioButtonComponent),
    multi: true
  }]
})
export class RadioButtonComponent implements ControlValueAccessor {
  @Input() options: { label: string, value: any }[] = [];
  value: any;

  onChange = (value: any) => {
// Intentionally left empty
};
  onTouched = () => {
// Intentionally left empty
};

  writeValue(value: any): void {
    this.value = value;
    this.onChange(value);
  }

  registerOnChange(fn: any): void {
    this.onChange = fn;
  }

  registerOnTouched(fn: any): void {
    this.onTouched = fn;
  }
}
