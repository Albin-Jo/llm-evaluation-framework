import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
@Component({
 selector: 'qrac-button',
 templateUrl: './qracbutton.component.html',
 styleUrls: ['./qracbutton.component.scss'],
 imports: [CommonModule]
})
export class QracButtonComponent implements OnInit {
  @Input() label = 'Button';
  @Input() type: 'primary' | 'secondary' | 'success' | 'tertiary' | 'danger' | 'warning' | 'info' = 'primary';
  @Input() size: 'small' | 'medium' | 'large' = 'medium'
  @Input() disabled = false;
  @Output() buttonClick = new EventEmitter<Event>();
  buttonClass = '';
  ngOnInit() {
    this.buttonClass = `button btn-${this.type} btn-${this.size}`;
  }
  onClick(event: Event) {
    if (!this.disabled) {
      this.buttonClick.emit(event);
    }
  }
 }
