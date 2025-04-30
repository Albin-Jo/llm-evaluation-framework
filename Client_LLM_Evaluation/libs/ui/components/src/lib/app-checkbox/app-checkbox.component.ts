import { Component, Input, forwardRef } from '@angular/core';
import { NG_VALUE_ACCESSOR, ControlValueAccessor, FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
@Component({
 selector: 'app-checkbox',
 templateUrl: './app-checkbox.component.html',
 styleUrls: ['./app-checkbox.component.scss'],
 imports: [CommonModule, FormsModule],
 providers: [{
   provide: NG_VALUE_ACCESSOR,
   useExisting: forwardRef(() => CheckboxComponent),
   multi: true
 }]
})
export class CheckboxComponent implements ControlValueAccessor {
 @Input() options: { label: string, value: any }[] = [];
 value: { [key: string]: boolean } = {};
 onChange = (value: any) => {
// Intentionally left empty
};
 onTouched = () => {
// Intentionally left empty
};
 writeValue(value: any): void {
   this.value = value;
   this.onChange(this.value);
 }
 registerOnChange(fn: any): void {
   this.onChange = fn;
 }
 registerOnTouched(fn: any): void {
   this.onTouched = fn;
 }
 handleChange(optionValue: any, event: Event): void {
   const inputElement = event.target as HTMLInputElement;
   const isChecked = inputElement.checked;
   if (isChecked) {
     this.value[optionValue] = true;
   } else {
     delete this.value[optionValue];
   }
   this.onChange(this.value);
   this.onTouched();
 }
}
