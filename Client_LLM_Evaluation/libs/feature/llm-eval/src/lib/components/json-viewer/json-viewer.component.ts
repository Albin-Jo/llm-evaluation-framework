/* Path: libs/feature/llm-eval/src/lib/components/json-viewer/json-viewer.component.ts */
import {
  Component,
  Input,
  OnChanges,
  SimpleChanges,
  ChangeDetectionStrategy,
  ChangeDetectorRef
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface TreeNode {
  key: string;
  value: any;
  type: string;
  expanded: boolean;
  children?: TreeNode[];
}

@Component({
  selector: 'app-json-viewer',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule
  ],
  templateUrl: './json-viewer.component.html',
  styleUrls: ['./json-viewer.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class JsonViewerComponent implements OnChanges {
  @Input() json: any = null;
  @Input() expanded = false;
  @Input() depth = 1;
  @Input() maxDepth = 20;
  @Input() collapsible = true;
  @Input() expandLevel = 1;

  treeData: TreeNode[] = [];
  searchText = '';
  searchResults: number = 0;

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['json']) {
      this.buildTree();
    }
  }

  /**
   * Build tree structure from JSON data
   */
  private buildTree(): void {
    if (this.json === null || this.json === undefined) {
      this.treeData = [];
      return;
    }

    try {
      let data = this.json;

      // If input is a string, try to parse it
      if (typeof this.json === 'string') {
        try {
          data = JSON.parse(this.json);
        } catch (e) {
          // If parsing fails, treat it as a string value
          this.treeData = [{
            key: 'root',
            value: this.json,
            type: 'string',
            expanded: true
          }];
          return;
        }
      }

      // Convert JSON to tree structure
      this.treeData = this.parseObject(data, 0);
      this.cdr.markForCheck();
    } catch (e) {
      console.error('Error building JSON tree:', e);
      this.treeData = [];
    }
  }

  /**
   * Parse any object into tree structure
   */
  private parseObject(obj: any, currentDepth: number): TreeNode[] {
    const result: TreeNode[] = [];

    // Stop recursion if max depth is reached
    if (currentDepth >= this.maxDepth) {
      return result;
    }

    // Handle null or undefined
    if (obj === null || obj === undefined) {
      return result;
    }

    // Handle array
    if (Array.isArray(obj)) {
      const arrayNode: TreeNode = {
        key: 'Array',
        value: `[${obj.length} items]`,
        type: 'array',
        expanded: currentDepth < this.expandLevel,
        children: []
      };

      obj.forEach((item, index) => {
        const valueType = this.getValueType(item);

        if (valueType === 'object' || valueType === 'array') {
          const childNode: TreeNode = {
            key: index.toString(),
            value: item,
            type: valueType,
            expanded: currentDepth + 1 < this.expandLevel,
            children: this.parseObject(item, currentDepth + 1)
          };
          arrayNode.children!.push(childNode);
        } else {
          arrayNode.children!.push({
            key: index.toString(),
            value: this.formatValue(item),
            type: valueType,
            expanded: false
          });
        }
      });

      result.push(arrayNode);
      return result;
    }

    // Handle object
    if (typeof obj === 'object') {
      Object.keys(obj).forEach(key => {
        const value = obj[key];
        const valueType = this.getValueType(value);

        if (valueType === 'object' || valueType === 'array') {
          // Complex type (object or array)
          const node: TreeNode = {
            key,
            value: value,
            type: valueType,
            expanded: currentDepth < this.expandLevel,
            children: this.parseObject(value, currentDepth + 1)
          };
          result.push(node);
        } else {
          // Simple type (string, number, boolean, null)
          result.push({
            key,
            value: this.formatValue(value),
            type: valueType,
            expanded: false
          });
        }
      });
    }

    return result;
  }

  /**
   * Get value type for display
   */
  private getValueType(value: any): string {
    if (value === null) return 'null';
    if (Array.isArray(value)) return 'array';

    return typeof value;
  }

  /**
   * Format value for display
   */
  private formatValue(value: any): string {
    if (value === null) return 'null';
    if (value === undefined) return 'undefined';

    if (typeof value === 'string') {
      return `"${value}"`;
    }

    return String(value);
  }

  /**
   * Toggle node expansion
   */
  toggleNode(node: TreeNode): void {
    if (this.collapsible) {
      node.expanded = !node.expanded;
      this.cdr.markForCheck();
    }
  }

  /**
   * Expand all nodes
   */
  expandAll(): void {
    this.toggleAll(true);
  }

  /**
   * Collapse all nodes
   */
  collapseAll(): void {
    this.toggleAll(false);
  }

  /**
   * Toggle all nodes to specified state
   */
  private toggleAll(expand: boolean): void {
    const toggleNode = (nodes: TreeNode[]) => {
      nodes.forEach(node => {
        node.expanded = expand;
        if (node.children) {
          toggleNode(node.children);
        }
      });
    };

    toggleNode(this.treeData);
    this.cdr.markForCheck();
  }

  /**
   * Search JSON for text
   */
  search(): void {
    if (!this.searchText) {
      this.resetSearch();
      return;
    }

    const searchText = this.searchText.toLowerCase();
    let count = 0;

    const searchNode = (nodes: TreeNode[]) => {
      nodes.forEach(node => {
        const keyMatch = node.key.toLowerCase().includes(searchText);
        const valueMatch = typeof node.value === 'string' && node.value.toLowerCase().includes(searchText);

        if (keyMatch || valueMatch) {
          count++;
          node.expanded = true;

          // Expand parent nodes
          let parent = node;
          while (parent && parent.children) {
            parent.expanded = true;
            // Not really traversing upward here, but expanding all encountered nodes
          }
        }

        if (node.children) {
          searchNode(node.children);
        }
      });
    };

    searchNode(this.treeData);
    this.searchResults = count;
    this.cdr.markForCheck();
  }

  /**
   * Reset search state
   */
  resetSearch(): void {
    this.searchText = '';
    this.searchResults = 0;
    this.buildTree(); // Rebuild tree to reset expansion state
  }

  /**
   * Copy JSON to clipboard
   */
  copyToClipboard(): void {
    try {
      const jsonString = JSON.stringify(this.json, null, 2);
      navigator.clipboard.writeText(jsonString)
        .then(() => {
          // Success
        })
        .catch(err => {
          console.error('Could not copy text: ', err);
        });
    } catch (e) {
      console.error('Error copying to clipboard:', e);
    }
  }

  /**
   * Get node name class
   */
  getKeyClass(node: TreeNode): string {
    return `json-key json-${node.type}-key`;
  }

  /**
   * Get value class
   */
  getValueClass(node: TreeNode): string {
    return `json-value json-${node.type}`;
  }
}