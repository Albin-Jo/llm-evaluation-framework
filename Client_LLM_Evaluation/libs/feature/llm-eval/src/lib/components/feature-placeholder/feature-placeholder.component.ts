/* Path: libs/feature/llm-eval/src/lib/components/feature-placeholder/feature-placeholder.component.ts */
import { Component, Input, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { QracButtonComponent } from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-feature-placeholder',
  standalone: true,
  imports: [CommonModule, QracButtonComponent],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './feature-placeholder.component.html',
  styleUrls: ['./feature-placeholder.component.scss']
})
export class FeaturePlaceholderComponent {
  @Input() featureTitle = 'Feature';
  @Input() featureIcon = '';
  @Input() description = 'This feature is coming soon.';

  constructor(private router: Router) {}

  navigateToDatasets(): void {
    this.router.navigate(['/llm-eval/datasets']);
  }
}
