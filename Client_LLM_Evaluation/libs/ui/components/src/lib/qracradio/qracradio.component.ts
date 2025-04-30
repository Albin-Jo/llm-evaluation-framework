import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule, ValidatorFn, AbstractControl, ValidationErrors } from '@angular/forms';
@Component({
 selector: 'qrac-radio',
 templateUrl: './qracradio.component.html',
 styleUrls: ['./qracradio.component.scss'],
 imports: [CommonModule, ReactiveFormsModule]
})
export class QracRadioComponent implements OnInit, OnDestroy, AfterViewInit {
  @Input() label = '';
  @Input() options: { value: string, label: string }[] = [];
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() disabled = false; // New input for disabling the radio buttons
  @Input() initialValue = '';
  @Output() valueChange = new EventEmitter<string>();
  control!: FormControl;
  ngOnInit() {
    this.control = new FormControl({ value: this.initialValue, disabled: this.disabled }, this.required ? Validators.required : null);
  }
  ngAfterViewInit() {
    // Perform any post-initialization tasks here
  }
  ngOnDestroy() {
    // Cleanup tasks when component is destroyed
  }
 }
