import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { QracButtonComponent } from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-placeholder',
  standalone: true,
  imports: [CommonModule, QracButtonComponent],
  template: `
    <div class="placeholder-container">
      <div class="card">
        <div class="card-header">
          <div class="card-title">{{ title }} Module</div>
        </div>
        <div class="card-content">
          <div class="development-notice">
            <div class="notice-icon">{{ icon }}</div>
            <div class="notice-title">{{ title }} Module Under Development</div>
            <div class="notice-message">This feature is currently under development and will be available soon.</div>
          </div>

          <div class="mt-4">
            <h3 class="feature-title">About {{ title }}</h3>
            <p>{{ description }}</p>
          </div>

          <div class="feature-list">
            <h3 class="feature-title">Planned Features</h3>
            <ul class="feature-items">
              <ng-container *ngIf="section === 'evaluations'">
                <li>Create and configure evaluation experiments</li>
                <li>Choose from multiple evaluation methods (RAGAS, DeepEval, etc.)</li>
                <li>View detailed evaluation results and metrics</li>
                <li>Export evaluation reports</li>
              </ng-container>

              <ng-container *ngIf="section === 'microagents'">
                <li>Configure microservice-based agents</li>
                <li>Monitor agent health and performance</li>
                <li>Customize agent behavior</li>
                <li>Test agents with sample inputs</li>
              </ng-container>

              <ng-container *ngIf="section === 'prompts'">
                <li>Create and manage prompt templates</li>
                <li>Test prompts with variables</li>
                <li>Organize prompts by category</li>
                <li>Version control for prompts</li>
              </ng-container>

              <ng-container *ngIf="section === 'comparisons'">
                <li>Compare multiple evaluations side-by-side</li>
                <li>Visualize performance differences</li>
                <li>Generate comparative reports</li>
                <li>Track improvements over time</li>
              </ng-container>

              <ng-container *ngIf="section === 'settings'">
                <li>Manage API connections</li>
                <li>Configure system preferences</li>
                <li>Manage user accounts and permissions</li>
                <li>Set up notification preferences</li>
              </ng-container>

              <ng-container *ngIf="!['evaluations', 'microagents', 'prompts', 'comparisons', 'settings'].includes(section)">
                <li>Feature planning in progress</li>
                <li>More details coming soon</li>
              </ng-container>
            </ul>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .placeholder-container {
      padding: 20px 0;
    }

    .card {
      background-color: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    }

    .card-header {
      padding: 15px 20px;
      border-bottom: 1px solid #f0f0f0;
    }

    .card-title {
      font-size: 18px;
      font-weight: 500;
    }

    .card-content {
      padding: 20px;
    }

    .development-notice {
      background-color: #f8f9fa;
      border: 1px dashed #ccc;
      border-radius: 8px;
      padding: 20px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      margin-bottom: 20px;
    }

    .notice-icon {
      font-size: 32px;
      margin-bottom: 15px;
    }

    .notice-title {
      font-size: 18px;
      font-weight: 500;
      margin-bottom: 10px;
    }

    .notice-message {
      font-size: 14px;
      color: #666;
    }

    .feature-title {
      font-size: 16px;
      font-weight: 500;
      margin-bottom: 10px;
    }

    .feature-list {
      margin-top: 20px;
    }

    .feature-items {
      list-style-type: disc;
      padding-left: 20px;

      li {
        margin-bottom: 5px;
      }
    }

    .mt-4 {
      margin-top: 20px;
    }
  `]
})
export class PlaceholderComponent {
  @Input() title = 'Feature';
  @Input() icon = '🚧';
  @Input() description = 'This feature is currently under development.';
  @Input() section = 'feature';
}
