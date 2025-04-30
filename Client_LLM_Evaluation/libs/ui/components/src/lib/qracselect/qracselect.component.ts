import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
@Component({
  selector: 'qrac-select',
  templateUrl: './qracselect.component.html',
  styleUrls: ['./qracselect.component.scss'],
  imports: [CommonModule, ReactiveFormsModule]
})
export class QracSelectComponent implements OnInit, OnDestroy, AfterViewInit {

  @Input() label = '';
  @Input() options: { value: string, label: string }[] = [];
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() initialValue = '';
  @Input() isFloat = false;
  @Input() infoText= '';
  @Output() valueChange = new EventEmitter<string>();
  @Output() keyPressEvent = new EventEmitter<KeyboardEvent>();
  form!: FormGroup;
  control!: FormControl;
  ngOnInit() {
    this.control = new FormControl(this.initialValue, this.required ? Validators.required : null);
    this.form = new FormGroup({
      select: this.control
    });
  }
  ngAfterViewInit() {
    // Perform any post-initialization tasks here
  }
  ngOnDestroy() {
    // Cleanup tasks when the component is destroyed
  }
  onKeyPress(event: KeyboardEvent) {
    this.keyPressEvent.emit(event);
  }
  get value() {
    return this.control!.value;
  }
  set value(val: string) {
    this.control!.setValue(val);
    this.valueChange.emit(val);
  }
 }
