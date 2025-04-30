/* Path: libs/feature/llm-eval/src/lib/pages/comparisons/comparisons.page.ts */
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FeaturePlaceholderComponent } from '../../components/feature-placeholder/feature-placeholder.component';

@Component({
  selector: 'app-comparisons',
  standalone: true,
  imports: [CommonModule, FeaturePlaceholderComponent],
  template: `
    <app-feature-placeholder
      featureTitle="Comparisons"
      featureIcon="comparisons.svg"
      description="Compare different RAG implementations and models side-by-side with detailed metrics.">
    </app-feature-placeholder>
  `
})
export class ComparisonsPage {}
