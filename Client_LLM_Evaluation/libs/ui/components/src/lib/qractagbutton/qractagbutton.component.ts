import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';

interface Filter {
    name: string;
    count: number | string;
    statusColor?: string;
    statusIcon?: string;
    isActive: boolean;
}
@Component({
 selector: 'qrac-tagbutton',
 templateUrl: './qractagbutton.component.html',
 styleUrls: ['./qractagbutton.component.scss'],
 imports: [CommonModule]
})
export class QracTagButtonComponent implements OnInit {
  @Input() label = 'Button';
  @Input() type: 'primary' | 'secondary' | 'success' | 'danger' | 'warning' | 'info' = 'primary';
  @Input() disabled = false;
  @Output() buttonClick = new EventEmitter<Event>();
  buttonClass = '';
  ngOnInit() {
    this.buttonClass = `btn btn-${this.type}`;
  }
  onClick(event: Event) {
    if (!this.disabled) {
      this.buttonClick.emit(event);
    }
  }

  filters: Filter[] = [
    { name: 'All', count: 100, isActive: true },
    { name: 'Completed', count: 10, isActive: false },

    { name: 'Submitted', count: 95, statusColor: 'var(--qrds-brand-color-success)', isActive: false },

    { name: 'In-Progress', count: (5).toString().padStart(2, '0'), statusColor: 'var(--qrds-brand-color-warning)', isActive: false },

    { name: 'Return', count: (5).toString().padStart(2, '0'), statusColor: 'var(--qrds-brand-color-danger)',isActive: false },

    { name: 'No PNR Found', count: (5).toString().padStart(2, '0'), statusIcon: '/assets/images/menu/warning.svg', isActive: false }
    // ... other filters
  ];

selectFilter(event: Event, selectedFilter: Filter) {
    console.log(selectedFilter)
    this.filters.forEach(filter => filter.isActive = false);
    selectedFilter.isActive = true;
    this.buttonClick.emit(event);
  }
 }
