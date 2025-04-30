import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit, SimpleChanges } from
'@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule, FormsModule  } from '@angular/forms';
@Component({
  selector: 'qrac-range',
  templateUrl: './qracrange.component.html',
  styleUrls: ['./qracrange.component.scss'],
  imports: [CommonModule, ReactiveFormsModule, FormsModule ]
})
export class QracRangetComponent implements OnInit, OnDestroy, AfterViewInit {

  @Input() label = '';
  @Input() min = 0;
  @Input() max = 10;
  @Input() inputValue: number = this.max/2;
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() initialValue = '';
  @Output() valueChange = new EventEmitter<number>();
  form!: FormGroup;
  control!: FormControl;

  ngOnInit() {
    this.control = new FormControl(this.inputValue, Validators.compose([this.required ? Validators.required : null]));
    this.form = new FormGroup({
      input: this.control
    });
  }

  ngAfterViewInit() {
    // Perform any post-initialization tasks here
  }
  ngOnDestroy() {
    // Cleanup tasks when the component is destroyed
  }


  ngOnChanges(changes: SimpleChanges) {
    if (changes['max']) {
      this.inputValue = this.max / 2;
    }
  }
  get value() {
    return this.control!.value;
  }
  set value(val: number) {
    this.control!.setValue(val);
    this.valueChange.emit(val);
  }
 }
