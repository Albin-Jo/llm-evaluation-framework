import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit } from
'@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
@Component({
  selector: 'qrac-toggle',
  templateUrl: './qractoggle.component.html',
  styleUrls: ['./qractoggle.component.scss'],
  imports: [CommonModule, ReactiveFormsModule]
})
export class QracTogglexComponent implements OnInit, OnDestroy, AfterViewInit {

  @Input() label = '';
  @Input() inputValue = '';
  @Input() placeholder = 'Enter Value';
  @Input() isMandatory = true;
  @Input() initialValue = false;
  @Input() disabled = false;
  @Output() valueChange = new EventEmitter<boolean>();
  @Output() keyPressEvent = new EventEmitter<KeyboardEvent>();
  form!: FormGroup;
  control!: FormControl;
  ngOnInit() {
    this.control = new FormControl({value: this.initialValue, disabled: this.disabled});
  }

  ngAfterViewInit() {
    // Perform any post-initialization tasks here
  }
  ngOnDestroy() {
    // Cleanup tasks when the component is destroyed
  }

 }
