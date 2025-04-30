import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule, ValidatorFn, AbstractControl, ValidationErrors } from '@angular/forms';
@Component({
 selector: 'qrac-checkbox',
 templateUrl: './qraccheckbox.component.html',
 styleUrls: ['./qraccheckbox.component.scss'],
 imports: [CommonModule, ReactiveFormsModule]
})
export class QracCheckboxComponent implements OnInit, OnDestroy, AfterViewInit {
  @Input() label = '';
  @Input() options: { value: string, label: string }[] = [];
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() disabled = false; // New input for disabling the checkboxes
  @Output() valueChange = new EventEmitter<string[]>();
  form!: FormGroup;
  controls: { [key: string]: FormControl } = {};
  ngOnInit() {
    this.form = new FormGroup({});
    this.options.forEach(option => {
      this.controls[option.value] = new FormControl({ value: false, disabled: this.disabled });
      this.form.addControl(option.value, this.controls[option.value]);
    });
    if (this.required) {
      this.form.setValidators(this.minSelectedCheckboxes(1));
    }
  }

  ngAfterViewInit() {
    // Perform any post-initialization tasks here
  }

  ngOnDestroy() {
    // Cleanup tasks when component is destroyed
  }

  onCheckboxChange(value: string) {
    this.valueChange.emit(this.getSelectedValues());
  }

  getSelectedValues(): string[] {
    return Object.keys(this.controls).filter(key => this.controls[key].value);
  }

  minSelectedCheckboxes(min: number): ValidatorFn {
    return (formGroup: AbstractControl): ValidationErrors | null => {
      const totalSelected = Object.keys((formGroup as FormGroup).controls)
        .map(key => (formGroup as FormGroup).controls[key].value)
        .reduce((prev, next) => next ? prev + 1 : prev, 0);
      return totalSelected >= min ? null : { required: true };
    };
  }
 }
