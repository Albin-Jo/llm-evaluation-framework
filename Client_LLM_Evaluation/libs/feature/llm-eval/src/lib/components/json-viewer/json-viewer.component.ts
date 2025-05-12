/* Path: libs/feature/llm-eval/src/lib/components/simple-json-viewer/simple-json-viewer.component.ts */
import { Component, Input, OnChanges, SimpleChanges, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-simple-json-viewer',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="json-viewer">
      <div class="json-viewer-header" *ngIf="showHeader">
        <div class="viewer-title">{{ title }}</div>
        <button class="copy-button" (click)="copyToClipboard()" title="Copy to clipboard">
          Copy
        </button>
      </div>
      <pre class="json-content">{{ formattedJson }}</pre>
    </div>
  `,
  styles: [`
    .json-viewer {
      width: 100%;
      border: 1px solid #ced4da;
      border-radius: 4px;
      background-color: #f8f9fa;
      overflow: hidden;
    }

    .json-viewer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      background-color: #f1f3f5;
      border-bottom: 1px solid #ced4da;
    }

    .viewer-title {
      font-weight: 500;
      font-size: 14px;
    }

    .copy-button {
      background: none;
      border: none;
      color: #0d6efd;
      cursor: pointer;
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 4px;
    }

    .copy-button:hover {
      background-color: rgba(13, 110, 253, 0.1);
    }

    .json-content {
      margin: 0;
      padding: 12px;
      font-family: monospace;
      font-size: 13px;
      white-space: pre-wrap;
      color: #333;
      max-height: 300px;
      overflow-y: auto;
    }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class SimpleJsonViewerComponent implements OnChanges {
  @Input() json: any = null;
  @Input() title: string = '';
  @Input() showHeader: boolean = false;

  formattedJson: string = '';

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['json']) {
      this.formatJson();
    }
  }

  formatJson(): void {
    try {
      if (this.json === null || this.json === undefined) {
        this.formattedJson = 'null';
        return;
      }

      if (typeof this.json === 'string') {
        try {
          // If the input is a JSON string, parse and format it
          const parsed = JSON.parse(this.json);
          this.formattedJson = JSON.stringify(parsed, null, 2);
        } catch (e) {
          // If parsing fails, treat it as a regular string
          this.formattedJson = this.json;
        }
      } else {
        // Format any other type of input
        this.formattedJson = JSON.stringify(this.json, null, 2);
      }
    } catch (e) {
      console.error('Error formatting JSON:', e);
      this.formattedJson = 'Error: Could not format JSON';
    }
  }

  copyToClipboard(): void {
    try {
      navigator.clipboard.writeText(this.formattedJson)
        .then(() => {
          // Success - could add a toast notification here
        })
        .catch(err => {
          console.error('Could not copy text: ', err);
        });
    } catch (e) {
      console.error('Error copying to clipboard:', e);
    }
  }
}